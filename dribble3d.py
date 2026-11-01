#!/usr/bin/env python3
"""dribble3d: closed-loop sketch / image -> 3D.

One file. Runs the same way on Kaggle (python dribble3d.py --tunnel) and Modal (modal_app.py imports `app`).

Pipeline (matches the report):
  Stage 1   adaptive silhouette + stroke extraction
  Stage 2   CLIP zero-shot label + local vision model (Qwen2-VL, no API) + keyword feedback gate
  Stage 3a  SDXL (fallback SD1.5) + ControlNet Scribble, best-of-N photoreal 2D candidates, u2net cutout, 0.85 framing (v1 settings)
  Stage 4   geometry-aware reranking  S = .40 EdgeIoU + .30 SilIoU + .15 CLIP + .15 Readiness3D
  Stage 3b  TripoSR triplane -> marching cubes mesh
  Stage 8   closed loop: 3D silhouette IoU vs the sketch, retry with next candidate, CPU tier-0 fallback
"""
from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.request
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi
from scipy import sparse
from scipy.spatial import cKDTree
from skimage import measure, morphology

try:
    import torch
except Exception:  # CPU-only tier-0 still works without torch
    torch = None

BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.environ.get("D3_OUT", os.path.join(BASE, "outputs"))
TSR_DIR = os.environ.get("TSR_DIR", os.path.join(BASE, "TripoSR"))
os.makedirs(OUT_DIR, exist_ok=True)

REMBG_MODEL = os.environ.get("D3_REMBG", "isnet-general-use")   # sharper object edges than u2net; falls back to u2net if it cannot load
SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN_ID = "lllyasviel/control_v11p_sd15_scribble"
# Photoreal 2D stage: SDXL + scribble ControlNet (1024 px). Falls back to the SD 1.5 pair above if it cannot load. D3_SD=15 forces SD 1.5.
SD_KIND = os.environ.get("D3_SD", "xl")
SDXL_ID = os.environ.get("D3_SDXL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
SDXL_CN = os.environ.get("D3_SDXL_CN", "xinsir/controlnet-scribble-sdxl-1.0")
SDXL_VAE = "madebyollin/sdxl-vae-fp16-fix"
SDXL_RES = int(os.environ.get("D3_SDXL_RES", "1024"))
CLIP_ID = os.environ.get("D3_CLIP_ID", "openai/clip-vit-large-patch14")   # ViT-L/14: far better on sketches than B/32
CLIP_FALLBACK = "openai/clip-vit-base-patch32"                           # used only if L/14 fails to load
# cosine range that maps to 0..1 in the 3D-readiness rerank (L/14 cosines sit lower than B/32)
CLIP_RANGE = {"openai/clip-vit-base-patch32": (0.18, 0.35), "openai/clip-vit-large-patch14": (0.12, 0.28)}
TSR_ID = "stabilityai/TripoSR"
HY_REPO = os.environ.get("D3_HY_REPO", "tencent/Hunyuan3D-2mini")
HY_SUB = os.environ.get("D3_HY_SUB", "hunyuan3d-dit-v2-mini")          # "hunyuan3d-dit-v2-mini-turbo" + D3_HY_TURBO=1 for speed
HY_TURBO = os.environ.get("D3_HY_TURBO", "0") == "1"
HY_STEPS = int(os.environ.get("D3_HY_STEPS", "24"))
HY_RES = int(os.environ.get("D3_HY_RES", "256"))                       # octree resolution
HY_DIR = os.environ.get("HY_DIR", os.path.join(BASE, "Hunyuan3D-2"))
BACKEND = "triposr"                                                    # v1 pipeline: TripoSR only

MV_ON = os.environ.get("D3_MV", "1") != "0"                       # allow the Multi-view option (D3_MV=0 hides it, nothing extra is downloaded)
MV_DEFAULT = MV_ON and os.environ.get("D3_MV_DEFAULT", "0") == "1"   # page opens with Multi-view selected (Modal sets 1)
ZPP_ID = os.environ.get("D3_ZPP_ID", "sudo-ai/zero123plus-v1.2")   # multi-view diffusion: one image -> 6 views
ZPP_PIPE = os.environ.get("D3_ZPP_PIPE", "sudo-ai/zero123plus-pipeline")
HYMV_REPO = os.environ.get("D3_HYMV_REPO", "tencent/Hunyuan3D-2mv")  # multi-view shape model: front + left + back -> mesh
HYMV_SUB = os.environ.get("D3_HYMV_SUB", "hunyuan3d-dit-v2-mv")
MV_STEPS = int(os.environ.get("D3_MV_STEPS", "75"))
HYMV_STEPS = int(os.environ.get("D3_HYMV_STEPS", "30"))
HYMV_RES = int(os.environ.get("D3_HYMV_RES", "256"))
MV_SWAP = os.environ.get("D3_MV_SWAP", "0") == "1"                # set to 1 if the left/right views come out mirrored
KEEP_JOBS = int(os.environ.get("D3_KEEP_JOBS", "16"))              # finished jobs kept on disk and in the result cache; older ones are deleted
LABEL_CONF = float(os.environ.get("D3_LABEL_CONF", "0.25"))       # below this CLIP confidence we warn that the label may be wrong

VLM_MODE = os.environ.get("D3_VLM", "1")             # 0 = CLIP only
VLM_ID = os.environ.get("D3_VLM_ID", "")             # "" = auto: Qwen2.5-VL-7B on big GPUs (>=30 GB), Qwen2-VL-2B otherwise
VLM_LOW = float(os.environ.get("D3_VLM_LOW", "0.05"))   # AI label is flagged only if CLIP finds it very unlikely

CANVAS = 512
N_CAND = int(os.environ.get("D3_NCAND", "3"))
FRAME = float(os.environ.get("D3_FRAME", "0.85"))   # v1: object fills 85% of the grey frame
MC_RES = int(os.environ.get("D3_MC_RES", "256"))
SMOOTH_ITERS = int(os.environ.get("D3_SMOOTH", "4"))
FACE_BUDGET = int(os.environ.get("D3_FACES", "250000"))
IOU_RETRY = float(os.environ.get("D3_IOU_RETRY", "0.60"))      # below this: try next candidate
IOU_FALLBACK = float(os.environ.get("D3_IOU_FALLBACK", "0.0"))   # 0 = never swap a neural mesh for the CPU blob (v1 behaviour)
W = dict(edge=0.40, sil=0.30, clip=0.15, ready=0.15)
# Fast / Quality only change candidate count and mesh detail. Both use SD 1.5 + ControlNet for the 2D render and TripoSR for 3D.
MODE_CFG = dict(fast=dict(backend="tsr", ncand=3, steps=26, res=256),
                quality=dict(backend="tsr", ncand=5, steps=36, res=320))

LABELS = {   # canonical label (goes into the SD prompt) -> names CLIP is asked about
    "apple": ["apple"], "pear": ["pear"], "banana": ["banana"], "mushroom": ["mushroom"], "strawberry": ["strawberry"],
    "pineapple": ["pineapple"], "carrot": ["carrot"], "cherry": ["cherries"], "watermelon": ["watermelon slice"],
    "chair": ["chair"], "sofa": ["sofa", "couch"], "table": ["table"], "bed": ["bed"], "lamp": ["desk lamp", "lamp"],
    "light bulb": ["light bulb"],
    "car": ["car"], "truck": ["truck"], "bus": ["bus"], "motorcycle": ["motorcycle"], "airplane": ["airplane"],
    "helicopter": ["helicopter"], "boat": ["boat", "sailboat"], "bicycle": ["bicycle"], "rocket": ["rocket", "spaceship"],
    "train": ["train"],
    "sword": ["sword"], "shield": ["shield"], "axe": ["axe"], "hammer": ["hammer"], "wrench": ["wrench"],
    "scissors": ["pair of scissors"], "knife": ["knife"], "pencil": ["pencil"], "fork": ["fork"], "spoon": ["spoon"],
    "bottle": ["bottle"], "mug": ["mug", "coffee cup", "cup"], "teapot": ["teapot"], "vase": ["vase"], "bowl": ["bowl"],
    "helmet": ["helmet"], "crown": ["crown"], "shoe": ["shoe", "sneaker"], "boot": ["boot"], "hat": ["hat"],
    "glasses": ["pair of glasses", "eyeglasses"], "glove": ["glove", "mitten"], "backpack": ["backpack"],
    "umbrella": ["umbrella"], "necklace": ["necklace"], "ring": ["ring", "wedding ring"], "headphones": ["headphones"],
    "guitar": ["guitar"], "violin": ["violin"], "drum": ["drum"], "camera": ["camera"], "phone": ["smartphone", "mobile phone"],
    "laptop": ["laptop"], "clock": ["clock", "wall clock"], "key": ["key"], "lock": ["padlock"], "book": ["book"],
    "dumbbell": ["dumbbell"], "basketball": ["basketball"], "trophy": ["trophy"], "bell": ["bell"], "anchor": ["anchor"],
    "candle": ["candle"], "gift box": ["gift box", "present"],
    "house": ["house"], "castle": ["castle"], "tower": ["tower"], "lighthouse": ["lighthouse"], "windmill": ["windmill"],
    "tent": ["tent"], "bridge": ["bridge"],
    "tree": ["tree"], "flower": ["flower"], "cactus": ["cactus"], "leaf": ["leaf"], "pumpkin": ["pumpkin"],
    "mountain": ["mountain"], "cloud": ["cloud"], "sun": ["sun"], "moon": ["crescent moon"], "star": ["star"], "heart": ["heart"],
    "cat": ["cat"], "dog": ["dog"], "bird": ["bird"], "fish": ["fish"], "duck": ["duck"], "rabbit": ["rabbit"], "bear": ["bear"],
    "elephant": ["elephant"], "horse": ["horse"], "turtle": ["turtle"], "dinosaur": ["dinosaur"], "snake": ["snake"],
    "spider": ["spider"], "whale": ["whale"], "shark": ["shark"], "frog": ["frog"], "crab": ["crab"], "butterfly": ["butterfly"],
    "owl": ["owl"], "penguin": ["penguin"], "pig": ["pig"], "cow": ["cow"], "lion": ["lion"], "monkey": ["monkey"],
    "giraffe": ["giraffe"], "octopus": ["octopus"], "bee": ["bee"],
    "robot": ["robot"], "skull": ["skull"], "ghost": ["ghost"], "snowman": ["snowman"], "hand": ["hand"], "foot": ["foot"],
    "face": ["face"], "wheel": ["wheel"],
    "donut": ["donut"], "pizza": ["pizza"], "cupcake": ["cupcake"], "ice cream": ["ice cream cone"], "cake": ["cake"],
    "cube": ["cube", "box"], "sphere": ["sphere", "ball"], "cylinder": ["cylinder"], "pyramid": ["pyramid"],
}

# ---- extended vocabulary (about 430 labels). CLIP only picks from names it is given, so more names = fewer forced wrong guesses.
_EXTRA = """ambulance, angel, ant, anvil, arm, asparagus, avocado, bandage, barn, barrel, baseball, baseball bat, basket, bat, bathtub,
battery, bench, binoculars, blender, bone, boomerang, bottle cap, bow tie, bracelet, brain, bread, broccoli, broom, bucket,
bulldozer, calculator, camel, campfire, cannon, canoe, cello, chandelier, chicken, church, clarinet, coconut, computer mouse,
cookie, cooler, corn, crayon, crocodile, cruise ship, deer, diamond, dishwasher, dolphin, door, dragon, dragonfly, dresser,
drone, eagle, ear, egg, eggplant, envelope, eraser, eye, fan, feather, fence, finger, fire extinguisher, fire hydrant,
firetruck, flamingo, flashlight, flip flops, flying saucer, fountain, fox, frying pan, game controller, globe, goat, golf club,
grapes, hamburger, harp, hedgehog, hippo, hockey puck, hockey stick, hot air balloon, hot dog, hourglass, igloo, jacket, jar,
jellyfish, joystick, kangaroo, kettle, ladder, ladybug, lantern, leg, lemon, lighter, lightning, lipstick, lizard, lobster,
lollipop, magnet, mailbox, marker, medal, megaphone, mermaid, microphone, microscope, microwave, mosquito, mouth, nail, nose,
onion, orange, oven, pagoda, paint can, paintbrush, palm tree, panda, pants, paper clip, parachute, parrot, pawn, peach,
peacock, peanut, piano, pickup truck, picture frame, pillar, pillow, pliers, police car, popsicle, pot, potato, purse, raccoon,
radio, rainbow, rake, remote control, rhinoceros, rooster, roller coaster, roller skates, saw, saxophone, school bus, scorpion,
screwdriver, seahorse, sheep, ship wheel, shorts, shovel, sink, skateboard, skyscraper, sleeping bag, snail, snowflake,
soccer ball, sock, speaker, speedboat, squirrel, stairs, starfish, statue, steak, stethoscope, stop sign, stove, streetlight,
submarine, suitcase, swan, sweater, swing set, syringe, t-shirt, taco, teacup, teddy bear, telescope, television, tiger,
toaster, toilet, tomato, tooth, toothbrush, toolbox, tractor, traffic cone, traffic light, trash can, treasure chest, trombone,
trumpet, van, washing machine, watering can, wheelbarrow, wine bottle, wine glass, wolf, wristwatch, zebra"""
for _n in _EXTRA.replace("\n", " ").split(","):
    _n = _n.strip()
    if _n:
        LABELS.setdefault(_n, [_n])
for _k, _v in {"motorcycle": ["motorbike"], "airplane": ["plane", "jet airplane"], "phone": ["telephone"],
               "cake": ["birthday cake"], "turtle": ["sea turtle"], "cube": ["cardboard box"], "mug": ["teacup"]}.items():
    LABELS[_k] = list(dict.fromkeys(LABELS[_k] + _v))
LABELS.pop("teacup", None)   # folded into mug
# Unbounded: add your own, one per line, "label|alias|alias", via D3_LABELS_FILE=/path/labels.txt
_lf = os.environ.get("D3_LABELS_FILE")
if _lf and os.path.exists(_lf):
    for _line in open(_lf, encoding="utf-8"):
        _p = [x.strip() for x in _line.split("|") if x.strip()]
        if _p:
            LABELS[_p[0]] = list(dict.fromkeys(LABELS.get(_p[0], []) + _p))
LABEL_NAMES = list(LABELS)

S = dict(device="cpu", label_top=[], label_conf=None, sd=None, clip=None, clip_proc=None, clip_id=None, label_emb=None, vlm=None, vlm_proc=None, vlm_id=None, rembg=None, tsr=None, hy=None, zpp=None, hymv=None, err={}, loaded=False)
_LOCK = threading.Lock()
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
PROGRESS: dict = {}
PARTIAL: dict = {}      # early results (identified label, generated images) the page shows while the 3D model is still building


# ============================================================ model loading
def _shim_mcubes():
    """TripoSR imports torchmcubes (needs a CUDA compile). Serve it from scikit-image instead."""
    def marching_cubes(vol, thresh):
        v = vol.detach().float().cpu().numpy()
        verts, faces, _, _ = measure.marching_cubes(v, level=float(thresh))
        return (torch.from_numpy(np.ascontiguousarray(verts[:, ::-1])).float(),
                torch.from_numpy(np.ascontiguousarray(faces)).long())
    m = types.ModuleType("torchmcubes")
    m.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = m
    for name in ("xatlas", "moderngl"):  # only used for texture baking, which we do not use
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)


def _load_tsr(dev):
    _shim_mcubes()
    if not os.path.isdir(TSR_DIR):
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/VAST-AI-Research/TripoSR.git", TSR_DIR])
    if TSR_DIR not in sys.path:
        sys.path.insert(0, TSR_DIR)
    from tsr.system import TSR
    m = TSR.from_pretrained(TSR_ID, config_name="config.yaml", weight_name="model.ckpt")
    m.renderer.set_chunk_size(131072)
    return m.to(dev).eval()


def _load_hy(dev):
    if not os.path.isdir(HY_DIR):
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git", HY_DIR])
    if HY_DIR not in sys.path:
        sys.path.insert(0, HY_DIR)
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(HY_REPO, subfolder=HY_SUB, use_safetensors=True, device=dev)
    if HY_TURBO:
        pipe.enable_flashvdm()
    return pipe


def load_models():
    if S["loaded"]:
        return
    dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    S["device"] = dev
    print(f"[device] {dev}")

    def step(name, fn):
        t = time.time()
        try:
            S[name] = fn()
            print(f"[{name} loaded] {time.time() - t:.1f}s")
        except Exception as e:  # keep going: tier-0 still works
            S["err"][name] = repr(e)[:400]
            print(f"[{name} FAILED] {repr(e)[:400]}")
            gc.collect()  # a half-loaded model must not keep GPU memory the other models need
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _rembg():
        from rembg import new_session
        try:
            return new_session(REMBG_MODEL)
        except Exception as e:
            print(f"[rembg {REMBG_MODEL} FAILED, using u2net] {repr(e)[:200]}")
            return new_session("u2net")

    def _clip():
        from transformers import CLIPModel, CLIPProcessor
        last = None
        for cid in dict.fromkeys([CLIP_ID, CLIP_FALLBACK]):
            try:
                S["clip_proc"] = CLIPProcessor.from_pretrained(cid)
                m = CLIPModel.from_pretrained(cid).half().to(dev).eval()
                S["clip_id"], S["label_emb"] = cid, None
                print(f"[clip] using {cid}")
                return m
            except Exception as e:
                last = e
                print(f"[clip {cid} FAILED] {repr(e)[:300]}")
                gc.collect()
        raise last

    def _vlm():
        if VLM_MODE == "0":
            return None
        from transformers import AutoModelForImageTextToText, AutoProcessor
        total = torch.cuda.get_device_properties(0).total_memory
        vid = VLM_ID or ("Qwen/Qwen2.5-VL-7B-Instruct" if total >= 30 * 2**30 else "Qwen/Qwen2-VL-2B-Instruct")
        dt = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        S["vlm_proc"] = AutoProcessor.from_pretrained(vid)
        try:
            m = AutoModelForImageTextToText.from_pretrained(vid, dtype=dt)
        except TypeError:
            m = AutoModelForImageTextToText.from_pretrained(vid, torch_dtype=dt)
        S["vlm_id"] = vid
        print(f"[vlm] using {vid}")
        return m.eval()                       # parked in CPU RAM; vlm_name() moves it to the GPU only while it works

    def _sd():
        if SD_KIND == "xl":
            try:
                from diffusers import (AutoencoderKL, ControlNetModel, EulerAncestralDiscreteScheduler,
                                       StableDiffusionXLControlNetPipeline)
                cn = ControlNetModel.from_pretrained(SDXL_CN, torch_dtype=torch.float16)
                vae = AutoencoderKL.from_pretrained(SDXL_VAE, torch_dtype=torch.float16)
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    SDXL_ID, controlnet=cn, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                pipe.set_progress_bar_config(disable=True)
                S["sd_kind"] = "xl"
                print("[sd] using SDXL + scribble ControlNet")
                return pipe.to(dev)
            except Exception as e:
                S["err"]["sdxl"] = repr(e)[:400]
                print(f"[sdxl FAILED, falling back to SD 1.5] {repr(e)[:400]}")
                gc.collect()
                torch.cuda.empty_cache()
        S["sd_kind"] = "15"
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
        cn = ControlNetModel.from_pretrained(CN_ID, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_ID, controlnet=cn, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        return pipe.to(dev)

    step("rembg", _rembg)
    if dev == "cuda":
        step("clip", _clip)
        step("sd", _sd)
        step("tsr", lambda: _load_tsr(dev))
        step("vlm", _vlm)                      # optional and last: if it does not fit, the essentials above are already loaded
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu] {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB after loading (need ~3 GiB free for generation)")
        if free < 3 * 2**30:
            print("[gpu] WARNING: not enough free GPU memory. Another process (e.g. an earlier cell) is probably holding it: "
                  "Run -> Restart Session, then run ONLY the tunnel cell.")
    S["loaded"] = True


def neural_ready():
    return S["device"] == "cuda" and S["sd"] is not None and S["rembg"] is not None and (S["hy"] is not None or S["tsr"] is not None)


def backend_chain(pref=None):
    pref = pref or BACKEND
    order = ["hy", "tsr"] if pref in ("hy", "hunyuan") else ["tsr", "hy"]
    return [b for b in order if S[b] is not None]


# ============================================================ small helpers
def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def norm_mask(mask, size=128, fill=0.9):
    """Crop to bbox, fit into size x size keeping aspect, centre. Makes IoU scale/shift invariant."""
    b = bbox(mask)
    out = np.zeros((size, size), bool)
    if b is None:
        return out
    y0, y1, x0, x1 = b
    crop = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    h, w = crop.shape
    s = fill * size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA) > 127
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = r
    return out


def keep_largest(mask):
    mask = mask.astype(bool)
    lab, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_fill_holes(mask)


def plain_bg(rgb, tol=18):
    """Colour of the background if the image border is one flat colour (studio render, white-paper photo), else None."""
    h, w = rgb.shape[:2]
    k = max(2, int(0.02 * min(h, w)))
    border = np.concatenate([rgb[:k].reshape(-1, 3), rgb[-k:].reshape(-1, 3), rgb[:, :k].reshape(-1, 3), rgb[:, -k:].reshape(-1, 3)]).astype(np.float32)
    return np.median(border, axis=0) if float(border.std(0).max()) < tol else None


def open_gaps(mask, rgb, gap_tol=18, min_frac=0.004, max_frac=0.6):
    """Cut-out that keeps real see-through gaps (between chair legs, under a backrest, handle loops, ...).
    keep_largest() fills every enclosed hole, and the matting network also tends to paint such gaps as object; both turn the gap into a
    solid slab that the 3D model then builds. Here, large regions inside the mask that have the background colour are removed again,
    and only small holes (segmentation noise) are filled. Only acts on plain backgrounds; falls back to the plain mask if it looks unsafe."""
    mask = mask.astype(bool)
    area = int(mask.sum())
    bg = plain_bg(rgb)
    if area < 200 or bg is None:
        return mask
    dist = np.abs(rgb.astype(np.float32) - bg[None, None, :]).max(2)
    gap = drop_small(mask & (dist < gap_tol), max(40, int(min_frac * area)))
    rest = mask & ~gap
    if gap.sum() > max_frac * area or rest.sum() < 100 or np.abs(np.median(rgb[rest].reshape(-1, 3), axis=0) - bg).max() < gap_tol:
        gap = np.zeros_like(gap)                    # the object itself is grey/white like the background: do not eat it
    out = mask & ~gap
    lab, n = ndi.label(out)
    if n > 1:
        out = lab == (1 + int(np.argmax(ndi.sum(out, lab, range(1, n + 1)))))
    holes = ndi.binary_fill_holes(out) & ~out
    hl, hn = ndi.label(holes)
    if hn:
        sizes = np.bincount(hl.ravel())
        small = sizes <= max(30, int(min_frac * area))
        small[0] = False
        out = out | small[hl]
    return out if out.sum() > 0.5 * area else mask


def largest_open(mask, hole_frac=0.004):
    """Largest connected piece of a cut-out, keeping its real see-through gaps open (the old v1 behaviour: the render's matte decides).
    Only tiny holes (matting noise, below hole_frac of the object) are filled."""
    mask = mask.astype(bool)
    lab, n = ndi.label(mask)
    if n > 1:
        mask = lab == (1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1)))))
    holes = ndi.binary_fill_holes(mask) & ~mask
    hl, hn = ndi.label(holes)
    if hn:
        sizes = np.bincount(hl.ravel())
        small = sizes <= max(30, int(hole_frac * mask.sum()))
        small[0] = False
        mask = mask | small[hl]
    return mask


def drop_small(mask, min_px):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def to_data_url(img, fmt="JPEG", maxside=320, q=84):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    img.thumbnail((maxside, maxside))
    buf = io.BytesIO()
    img.save(buf, fmt, quality=q)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def load_rgb(b):
    im = Image.open(io.BytesIO(b))
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    im = im.convert("RGB")
    if max(im.size) > 1536:
        im.thumbnail((1536, 1536))
    return im


def detect_kind(rgb):
    a = np.array(rgb)
    g = a.min(axis=2)
    sat = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 1].mean()
    return "sketch" if ((g > 200).mean() > 0.80 and sat < 60) else "photo"


# ============================================================ stage 1: silhouette + strokes
def fill_silhouette(strokes):
    """Progressive elliptical closing until the outline encloses something, convex hull as last resort."""
    st = strokes.astype(np.uint8) * 255
    area = int(strokes.sum())
    pts = cv2.findNonZero(st)
    hull = cv2.convexHull(pts)
    hull_mask = np.zeros_like(st)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    if area / max(1, int((hull_mask > 0).sum())) > 0.70:     # already a solid blob
        return ndi.binary_fill_holes(st > 0)
    for k in (3, 7, 11, 17, 25, 35, 49):
        closed = cv2.morphologyEx(st, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        filled = ndi.binary_fill_holes(closed > 0)
        if filled.sum() > 1.15 * area:
            return drop_small(filled, max(30, int(0.003 * filled.size)))
    return hull_mask > 0


def extract_sketch(rgb):
    g = np.array(rgb).min(axis=2)
    if (g < 170).mean() < 0.0008:
        raise ValueError("The canvas looks empty. Draw something first.")
    h, w = g.shape
    bg = cv2.GaussianBlur(g, (0, 0), max(h, w) / 10)                   # shadow-invariant: I / blur(I)
    norm = np.clip(g.astype(np.float32) / (bg.astype(np.float32) + 1.0) * 255.0, 0, 255).astype(np.uint8)
    _, st = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    st = drop_small(st > 0, 24)
    if st.sum() < 40:
        raise ValueError("Could not find any strokes. Try drawing darker or thicker lines.")
    return st, fill_silhouette(st)


def stroke_color(rgb, strokes):
    """Nearest named colour of the pen colour used in the drawing, or None for plain black/grey ink."""
    a = np.array(rgb)[strokes]
    if len(a) < 30:
        return None
    hsv = cv2.cvtColor(a.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    colored = hsv[:, 1] > 80
    if colored.mean() < 0.35:
        return None
    med = np.median(a[colored].astype(np.float32), axis=0)
    names = list(COLOR_RGB)
    d = [float(np.sum((np.array(COLOR_RGB[n], np.float32) - med) ** 2)) for n in names]
    return names[int(np.argmin(d))]


def to_canvas(strokes, sil, size=CANVAS, fill=0.80):
    """Re-frame strokes + silhouette identically into a size x size canvas (what ControlNet sees)."""
    y0, y1, x0, x1 = bbox(strokes | sil)
    h, w = y1 - y0, x1 - x0
    s = fill * size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    oy, ox = (size - nh) // 2, (size - nw) // 2

    def tf(m, thr):
        c = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8) * 255, (nw, nh), interpolation=cv2.INTER_AREA) > thr
        out = np.zeros((size, size), bool)
        out[oy:oy + nh, ox:ox + nw] = c
        return out

    return tf(strokes, 40), tf(sil, 127)


def sketch_views(strokes512):
    thick = cv2.dilate(strokes512.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    control = Image.fromarray(np.stack([thick] * 3, -1))                   # white strokes on black
    paper = Image.fromarray(np.stack([255 - thick] * 3, -1))               # black strokes on white
    return control, paper


# ============================================================ stage 2: label + keyword feedback gate
FB_RULES = [
    (r"\b(thin|thinner|slim|slender|skinny)\w*", dict(add="slim, slender", z=0.75)),
    (r"\b(chunk|thick|fat|bulky|stout|puffy|round)\w*", dict(add="chunky, thick, rounded", z=1.3)),
    (r"\b(wood|wooden|carved)\w*", dict(add="carved wooden, wood grain")),
    (r"\b(metal|metallic|steel|chrome|silver)\w*", dict(add="polished metal, metallic sheen")),
    (r"\bgold\w*", dict(add="polished gold", preset="gold")),
    (r"\b(clay|plasticine)\w*", dict(add="matte clay sculpture", preset="clay")),
    (r"\b(terracotta|ceramic|pottery)\w*", dict(add="glazed ceramic", preset="terracotta")),
    (r"\b(toy|plastic|collectible|cartoon)\w*", dict(add="collectible vinyl toy, smooth plastic", preset="toy")),
    (r"\b(stone|statue|marble)\w*", dict(add="carved stone statue")),
    (r"\b(realistic|photoreal)\w*", dict(add="photorealistic, detailed")),
    (r"\b(exact|precise|faithful|follow)\w*", dict(cn=1.15)),
    (r"\b(loose|creative|freeform|artistic)\w*", dict(cn=0.65)),
]


COLOR_RGB = dict(red=(205, 48, 48), orange=(240, 140, 40), yellow=(244, 206, 60), green=(70, 160, 80), blue=(52, 110, 220),
                 purple=(130, 80, 190), pink=(240, 130, 170), brown=(130, 85, 55), black=(40, 40, 44), white=(236, 236, 232),
                 teal=(40, 160, 160), navy=(30, 50, 120))
COLOR_RE = re.compile(r"\b(" + "|".join(COLOR_RGB) + r")\b")


def parse_feedback(text):
    out = dict(add=[], z=1.0, cn=0.85, preset=None, hits=[], color=None)
    t = (text or "").lower()
    for pat, eff in FB_RULES:
        m = re.search(pat, t)
        if not m:
            continue
        out["hits"].append(m.group(0))
        if "add" in eff:
            out["add"].append(eff["add"])
        out["z"] *= eff.get("z", 1.0)
        out["cn"] = eff.get("cn", out["cn"])
        out["preset"] = eff.get("preset", out["preset"])
    cm = COLOR_RE.search(t)
    if cm:
        out["color"] = cm.group(1)
        out["hits"].append(cm.group(1))
    return out


def _clip_forward(images, texts):
    inp = S["clip_proc"](text=texts, images=images, return_tensors="pt", padding=True)
    dt = next(S["clip"].parameters()).dtype
    inp = {k: (v.to(S["device"], dt) if v.is_floating_point() else v.to(S["device"])) for k, v in inp.items()}
    with torch.no_grad():
        return S["clip"](**inp)


def _unit(t):
    t = t.float()
    return t / t.norm(dim=-1, keepdim=True)


def clip_cos(images, texts):
    o = _clip_forward(images, texts)
    return (_unit(o.image_embeds) @ _unit(o.text_embeds).T).cpu().numpy()


def clip_image_emb(images):
    return _unit(_clip_forward(images, ["a"]).image_embeds)


def clip_text_emb(texts, bs=200):
    blank = Image.new("RGB", (224, 224), (255, 255, 255))
    return torch.cat([_unit(_clip_forward([blank], texts[i:i + bs]).text_embeds) for i in range(0, len(texts), bs)])


# Prompt ensemble: every label is described several ways and the text embeddings are averaged (standard CLIP trick, +several % accuracy).
CLIP_TEMPLATES = [
    "a black and white line drawing of a {}.",
    "a hand-drawn sketch of a {}.",
    "a simple doodle of a {}.",
    "a pencil drawing of a {}.",
    "a child's drawing of a {}.",
    "an outline sketch of a single {}.",
    "a rough sketch of a {} on white paper.",
]


def label_bank():
    """(n_labels, d) unit text embeddings, computed once per model and cached."""
    if S.get("label_emb") is not None:
        return S["label_emb"]
    texts, owner = [], []
    for li, lab in enumerate(LABEL_NAMES):
        for nm in LABELS[lab]:
            for t in CLIP_TEMPLATES:
                texts.append(t.format(nm))
                owner.append(li)
    emb = clip_text_emb(texts)
    bank = torch.zeros(len(LABEL_NAMES), emb.shape[1], device=emb.device)
    bank.index_add_(0, torch.tensor(owner, device=emb.device), emb)
    S["label_emb"] = _unit(bank)
    return S["label_emb"]


def clip_views(strokes512, sil512):
    """What CLIP sees. The old code fed the whole 512 canvas with 1-2 px strokes; CLIP shrinks that to 224 px and the lines
    almost vanish. Here: tight square crop around the drawing, strokes thickened to ~3 px at CLIP's resolution,
    plus a filled-silhouette view and a mirrored view, and the three embeddings are averaged."""
    m = strokes512 | sil512
    ys, xs = np.nonzero(m)
    cy, cx = int(round((ys.min() + ys.max()) / 2)), int(round((xs.min() + xs.max()) / 2))
    side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.16) + 8
    R = 448

    def crop(mask):
        a = np.pad(mask.astype(np.uint8) * 255, side)
        y0, x0 = cy + side - side // 2, cx + side - side // 2
        return cv2.resize(a[y0:y0 + side, x0:x0 + side], (R, R), interpolation=cv2.INTER_AREA)

    st = (crop(strokes512) > 20).astype(np.uint8) * 255
    st = cv2.dilate(st, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    sl = crop(sil512) > 127
    line = 255 - st
    fill = np.minimum(np.where(sl, 205, 255).astype(np.uint8), line)
    return [Image.fromarray(np.stack([v] * 3, -1)) for v in (line, fill, np.ascontiguousarray(line[:, ::-1]))]


def guess_label(strokes512, sil512):
    """CLIP zero-shot label from the cleaned strokes. Keeps the top 5 (with probabilities) for the page."""
    if S["clip"] is None:
        return "object"
    try:
        bank = label_bank()
        img = _unit(clip_image_emb(clip_views(strokes512, sil512)).mean(0, keepdim=True))
        cos = (img @ bank.T)[0].float().cpu().numpy()
    except Exception as e:
        print(f"[label guess failed] {repr(e)[:300]}")
        S["label_conf"], S["label_top"] = None, []
        return "object"
    S["_cos"], S["_img"] = cos, img
    z = 100.0 * (cos - cos.max())
    p = np.exp(z) / np.exp(z).sum()
    order = np.argsort(-p)[:5]
    S["label_conf"] = float(p[order[0]])
    S["label_top"] = [(LABEL_NAMES[int(i)], float(p[int(i)])) for i in order]
    return LABEL_NAMES[int(order[0])]


# ============================================================ AI vision model names the object (open vocabulary), CLIP double-checks it
VLM_PROMPT = ("This is a rough hand-drawn sketch of one object. It may be an animal, a vehicle, a tool, food, a toy, a building or a "
              "character. What is it? Reply with only the object name, 1 to 3 words, lowercase, no punctuation. "
              "If you are unsure, give your best guess.")


def vlm_name(img):
    proc, m = S["vlm_proc"], S["vlm"]
    m.to(S["device"])
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VLM_PROMPT}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[img], return_tensors="pt")
    inp = {k: (v.to(S["device"], m.dtype) if v.is_floating_point() else v.to(S["device"])) for k, v in inp.items()}
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=12, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def clean_label(raw):
    t = re.sub(r"[^a-z0-9\- ]+", " ", ((raw or "").lower().strip().split("\n") or [""])[0])
    t = " ".join(t.split())
    while True:
        n = re.sub(r"^(this is|it is|it s|that is|this looks like|looks like|a|an|the|sketch of|drawing of|picture of|doodle of)\s+", "", t)
        if n == t:
            break
        t = n
    t = " ".join(t.split()[:4])
    return "" if t in ("", "unknown", "object", "sketch", "drawing", "none", "thing") else t


def _alias_map():
    m = {}
    for lab, names in LABELS.items():
        m[lab] = lab
        for nm in names:
            m.setdefault(nm, lab)
    return m


def _name_cos(v, canon, img):
    if canon:
        return float(S["_cos"][LABEL_NAMES.index(canon)])
    t = _unit(clip_text_emb([tp.format(v) for tp in CLIP_TEMPLATES]).mean(0, keepdim=True))
    return float((img @ t.T)[0, 0])


def label_is_low():
    c = S.get("label_conf")
    if c is None:
        return False
    return c < (VLM_LOW if str(S.get("label_source", "")).startswith("AI") else LABEL_CONF)


def identify_label(strokes512, sil512):
    """1) CLIP scores the whole vocabulary. 2) The vision model names the object in free words, on two renderings.
    3) If it gave two different names, CLIP picks the one that fits the drawing better. 4) CLIP's probability for the chosen
    name decides the 'not sure' flag; the page shows the AI name first, then CLIP's best alternatives."""
    S["label_source"], S["ai_note"], S["_cos"], S["_img"] = None, None, None, None
    clip_label = guess_label(strokes512, sil512)
    S["label_source"] = "CLIP guess"
    if S.get("vlm") is None:
        if VLM_MODE != "0":
            S["ai_note"] = (S["err"].get("vlm") or "model not loaded")[:90]
        return clip_label
    try:
        views = clip_views(strokes512, sil512)
        names = []
        for im in views[:2]:                          # line drawing + filled silhouette
            n = clean_label(vlm_name(im))
            if n and n not in names:
                names.append(n)
        if not names:
            raise ValueError("empty answer")
    except Exception as e:
        print(f"[vlm failed] {repr(e)[:300]}")
        S["ai_note"] = repr(e)[:90]
        _park(S["vlm"], "vlm")
        return clip_label
    _park(S["vlm"], "vlm")                             # unload the vision model: the 3D models get the GPU
    amap = _alias_map()
    names = [(n[:-1] if (n not in amap and n.endswith("s") and n[:-1] in amap) else n) for n in names]
    if S.get("_cos") is None:                          # CLIP scoring failed earlier: trust the AI alone
        S["label_source"], S["label_conf"], S["label_top"] = "AI vision model", None, []
        return names[0]
    img, cos = S["_img"], S["_cos"]
    clip_top3 = {l for l, _ in S["label_top"][:3]}
    scored = [(_name_cos(n, amap.get(n), img), n) for n in names]
    cv, v = max(scored)
    canon = amap.get(v)
    allcos = cos if canon else np.append(cos, cv)
    vi = LABEL_NAMES.index(canon) if canon else len(LABEL_NAMES)
    labels = LABEL_NAMES + ([] if canon else [v])
    p = np.exp(100.0 * (allcos - allcos.max()))
    p /= p.sum()
    rest = [int(i) for i in np.argsort(-p) if int(i) != vi][:4]
    S["label_top"] = [(v, float(p[vi]))] + [(labels[i], float(p[i])) for i in rest]
    S["label_conf"] = float(p[vi])
    agree = (canon in clip_top3) or (v in clip_top3)
    S["label_source"] = "AI vision model · CLIP agrees" if agree else "AI vision model"
    return v


# ============================================================ LLM assistant (local vision model; the Anthropic API path is optional)
# ============================================================ stage 3a: candidates, cutout, framing
def gen_candidates(control, prompt, neg, cn_scale, seed, n_cand=N_CAND, steps=22):
    pipe = S["sd"]
    xl = S.get("sd_kind") == "xl"

    def run(n, sd_seed):
        g = torch.Generator(device="cuda").manual_seed(sd_seed)
        if xl:
            big = control.resize((SDXL_RES, SDXL_RES), Image.LANCZOS)
            return pipe(prompt, negative_prompt=neg, image=big, num_inference_steps=max(steps, 28), guidance_scale=6.0,
                        controlnet_conditioning_scale=float(cn_scale), num_images_per_prompt=n, generator=g,
                        height=SDXL_RES, width=SDXL_RES).images
        return pipe(prompt, negative_prompt=neg, image=control, num_inference_steps=steps, guidance_scale=7.5,
                    controlnet_conditioning_scale=float(cn_scale), num_images_per_prompt=n, generator=g,
                    height=CANVAS, width=CANVAS).images

    try:
        return run(n_cand, seed)
    except RuntimeError as e:  # CUDA OOM: retry one image at a time
        print(f"[2D synthesis failed, retrying one candidate at a time] {repr(e)[:300]}")
        gc.collect()
        torch.cuda.empty_cache()
        out = []
        for k in range(n_cand):
            out += run(1, seed + k)
        return out


def refine_cutout(rgb, rgba):
    """Background removal clean-up on top of the matting network. On a plain background (generated renders, Zero123++ views, photos on
    paper) every pixel that has the background colour and is connected to the image border is removed, enclosed see-through gaps are
    re-opened, and the edge pixels (which are blended with the old background) get the colour of the nearest interior pixel."""
    a = rgba[..., 3] > 127
    bg = plain_bg(rgb)
    if bg is None or a.sum() < 200:
        return rgba
    near = np.abs(rgb.astype(np.float32) - bg[None, None, :]).max(2) < 30
    lab, _ = ndi.label(near)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    keyed = a & ~np.isin(lab, edge) if len(edge) else a
    if keyed.sum() < 0.65 * a.sum():                 # it would eat a big part of the object (object colour == background colour): skip the key
        keyed = a
    keyed = open_gaps(keyed, rgb)
    out = rgba.copy()
    out[..., 3] = keyed.astype(np.uint8) * 255
    out[..., :3] = edge_colors(rgb, keyed)
    return out


def edge_colors(rgb, mask):
    """RGB with the 2 px band along the cut-out edge replaced by the nearest interior colour (no grey/white fringe on the 3D model)."""
    core = ndi.binary_erosion(mask, iterations=2)
    if core.sum() < 50:
        return rgb
    _, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    out = rgb.copy()
    band = mask & ~core
    out[band] = rgb[iy, ix][band]
    return out


def segment(img):
    from rembg import remove
    rgb = img.convert("RGB")
    out = np.array(remove(rgb, session=S["rembg"]).convert("RGBA"))
    try:
        out = refine_cutout(np.array(rgb), out)
    except Exception as e:
        print("[cutout refine skipped]", repr(e)[:200])
    return out


def frame_fg(rgb_arr, mask, ratio=FRAME, size=CANVAS):
    y0, y1, x0, x1 = bbox(mask)
    crop = rgb_arr[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1].astype(np.float32)
    h, w = m.shape
    s = ratio * size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)[..., None]
    out = np.full((size, size, 3), 127.0, np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = out[oy:oy + nh, ox:ox + nw] * (1 - m) + crop * m
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# ============================================================ multi-view (optional): Zero123++ views -> Hunyuan3D-2mv
# Both models are loaded lazily on the first multi-view run, kept in CPU RAM, and moved to the GPU only while they work.
def _load_zpp():
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    p = DiffusionPipeline.from_pretrained(ZPP_ID, custom_pipeline=ZPP_PIPE, trust_remote_code=True, torch_dtype=torch.float16)
    p.scheduler = EulerAncestralDiscreteScheduler.from_config(p.scheduler.config, timestep_spacing="trailing")
    try:
        p.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return p


def _load_hymv():
    if not os.path.isdir(HY_DIR):
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git", HY_DIR])
    if HY_DIR not in sys.path:
        sys.path.insert(0, HY_DIR)
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(HYMV_REPO, subfolder=HYMV_SUB, use_safetensors=True, variant="fp16", device="cpu")


def mv_prefetch():
    """Download and load both multi-view models (CPU only), print the result, free them. Run once before using Multi-view."""
    for name, fn in (("zpp", _load_zpp), ("hymv", _load_hymv)):
        t = time.time()
        try:
            m = fn()
            print(f"[mv {name} OK] {time.time() - t:.0f}s")
            del m
        except Exception as e:
            print(f"[mv {name} FAILED] {repr(e)[:500]}")
        gc.collect()


def _tile_grid(grid):
    """Zero123++ returns a 2-column x 3-row grid, row-major. Relative azimuths 30, 90 / 150, 210 / 270, 330 degrees."""
    W, H = grid.size
    tw, th = W // 2, H // 3
    if abs(tw - th) > 4:
        raise RuntimeError(f"unexpected Zero123++ output size {W}x{H}")
    return [grid.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th)) for r in range(3) for c in range(2)]


def _park(pipe, key=None):
    """Move a pipeline back to CPU RAM and release cached GPU memory. If the move fails, drop the model so the GPU is not left holding it."""
    try:
        pipe.to("cpu")
    except Exception as e:
        print("[park failed, dropping the model]", repr(e)[:200])
        if key:
            S[key] = None
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mem():
    """One status line: GPU allocated/reserved, this process's RAM, job folders on disk. Printed after every run so any growth is visible."""
    g = ""
    if torch is not None and torch.cuda.is_available():
        g = f"gpu {torch.cuda.memory_allocated() / 2**30:.1f} alloc / {torch.cuda.memory_reserved() / 2**30:.1f} reserved GiB | "
    try:
        rss = int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) / 2**20
    except Exception:
        rss = -1.0
    try:
        n = len([d for d in os.listdir(OUT_DIR) if re.fullmatch(r"[a-f0-9]{12}", d)])
    except Exception:
        n = -1
    return f"[mem] {g}ram {rss:.1f} GiB | {n} job folders"


def _prune_outputs(keep=None):
    """Every run writes .glb/.obj/.stl/.ply files. Without this the folder grows until the Kaggle disk is full."""
    keep = KEEP_JOBS if keep is None else keep
    try:
        import shutil
        dirs = [os.path.join(OUT_DIR, d) for d in os.listdir(OUT_DIR) if re.fullmatch(r"[a-f0-9]{12}", d)]
        dirs.sort(key=os.path.getmtime, reverse=True)
        for d in dirs[keep:]:
            shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        print("[prune skipped]", repr(e)[:120])


def mv_generate(cutout, seed):
    """winning candidate (RGBA) -> six 320x320 views from Zero123++."""
    if S["zpp"] is None:
        t = time.time()
        S["zpp"] = _load_zpp()                                        # parked in CPU RAM between runs: no reload, no repeated 5 GB allocations
        print(f"[zero123++ loaded] {time.time() - t:.1f}s")           # (diffusers prints a harmless fp16/cpu warning when it is parked)
    pipe = S["zpp"]
    pipe.to("cuda")
    bg = Image.new("RGB", cutout.size, (255, 255, 255))
    bg.paste(cutout.convert("RGBA"), (0, 0), cutout.convert("RGBA"))
    cond = bg.resize((320, 320), Image.LANCZOS)
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            grid = pipe(cond, num_inference_steps=MV_STEPS).images[0]
    finally:
        _park(pipe, "zpp")
    return _tile_grid(grid)


def norm_views(rgbas, size=512, hfrac=0.80):
    """Put every view on one common scale. Object height is the same from every side, so match the bbox height, then shrink all
    views together if the widest one would not fit. Returns RGBA PIL images, or None for an empty view."""
    boxes, crops = [], []
    for a in rgbas:
        a = np.asarray(a)
        b = bbox(a[..., 3] > 127)
        if b is None or (a[..., 3] > 127).sum() < 50:
            return None
        y0, y1, x0, x1 = b
        boxes.append((y1 - y0, x1 - x0))
        crops.append(a[y0:y1, x0:x1])
    scales = [hfrac * size / h for h, _ in boxes]
    g = min(1.0, 0.95 * size / max(w * s for (_, w), s in zip(boxes, scales)))
    out = []
    for c, (h, w), s in zip(crops, boxes, scales):
        s *= g
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        canvas = np.zeros((size, size, 4), np.uint8)
        oy, ox = (size - nh) // 2, (size - nw) // 2
        canvas[oy:oy + nh, ox:ox + nw] = c
        out.append(Image.fromarray(canvas, "RGBA"))
    return out


def mv_views(cutout, tiles):
    """Cut out the Zero123++ tiles and pick left/back. Returns (dict for Hunyuan3D-2mv, preview list for the page)."""
    with ThreadPoolExecutor(len(tiles)) as ex:
        tile_rgba = list(ex.map(segment, tiles))
    left_i, back_i = (4, 3) if MV_SWAP else (1, 3)                    # relative azimuth 270 / 90 deg for the side, 210 deg for the back
    sel = [np.array(cutout.convert("RGBA")), tile_rgba[left_i], tile_rgba[back_i]]
    norm = norm_views(sel)
    if norm is None:
        raise RuntimeError("a generated view was empty after cutting out the background")
    names = ["Front (your image)"] + [f"Zero123++ {a}\u00b0" for a in (30, 90, 150, 210, 270, 330)]
    used = {0, 1 + left_i, 1 + back_i}
    def on_light(im):
        rgba = Image.fromarray(np.asarray(im)).convert("RGBA") if not isinstance(im, Image.Image) else im.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, (242, 242, 242, 255)), rgba).convert("RGB")
    shown = [on_light(cutout)] + [on_light(Image.fromarray(t)) for t in tile_rgba]
    preview = [dict(img=to_data_url(im, maxside=240), name=n, used=(i in used)) for i, (im, n) in enumerate(zip(shown, names))]
    return dict(front=norm[0], left=norm[1], back=norm[2]), preview


def reconstruct_hymv(views, seed):
    if S["hymv"] is None:
        t = time.time()
        S["hymv"] = _load_hymv()
        print(f"[hunyuan3d-2mv loaded] {time.time() - t:.1f}s")
    pipe = S["hymv"]
    pipe.to("cuda")
    try:
        try:
            out = pipe(image=views, num_inference_steps=HYMV_STEPS, octree_resolution=HYMV_RES, num_chunks=20000,
                       generator=torch.manual_seed(seed), output_type="trimesh")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out = pipe(image=views, num_inference_steps=HYMV_STEPS, octree_resolution=192, num_chunks=8000,
                       generator=torch.manual_seed(seed), output_type="trimesh")
    finally:
        _park(pipe, "hymv")
    return out[0] if isinstance(out, (list, tuple)) else out


# ============================================================ stage 4: geometry-aware scoring
def edge_iou(strokes512, cand_rgb, cand_mask):
    gray = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(gray, 60, 140)
    m8 = cand_mask.astype(np.uint8) * 255
    outline = cv2.morphologyEx(m8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    e = cv2.bitwise_or(e, outline)
    e = cv2.bitwise_and(e, cv2.dilate(m8, np.ones((7, 7), np.uint8)))
    skel = morphology.skeletonize(strokes512 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return iou(cv2.dilate(e, k) > 0, cv2.dilate(skel, k) > 0)


def readiness(cand_rgb, mask):
    """Does it look like something a single-image 3D model can use? shading variance + compact solid blob."""
    if mask.sum() < 50:
        return 0.0
    lum = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)[mask]
    shade = float(np.clip(lum.std() / 50.0, 0, 1))
    pts = cv2.findNonZero(mask.astype(np.uint8))
    hull = cv2.contourArea(cv2.convexHull(pts))
    solidity = float(mask.sum() / hull) if hull > 0 else 0.0
    return 0.7 * shade + 0.3 * min(1.0, solidity)


def score_candidate(strokes512, sil512, cand_rgb, mask, label):
    parts = dict(edge=edge_iou(strokes512, cand_rgb, mask), sil=iou(norm_mask(mask), norm_mask(sil512)),
                 clip=0.0, ready=readiness(cand_rgb, mask))
    if S["clip"] is not None:
        cos = float(clip_cos([Image.fromarray(cand_rgb)], [f"a 3d render of a {label}"])[0][0])
        lo, hi = CLIP_RANGE.get(S.get("clip_id"), (0.14, 0.30))
        parts["clip"] = float(np.clip((cos - lo) / (hi - lo), 0, 1))
    total = sum(W[k] * parts[k] for k in W)
    return total, parts


# ============================================================ stage 3b: reconstruction + alignment
def reconstruct_tsr(framed, res=MC_RES):
    with torch.no_grad():
        codes = S["tsr"]([framed], device=S["device"])
        try:
            return S["tsr"].extract_mesh(codes, has_vertex_color=True, resolution=res)[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return S["tsr"].extract_mesh(codes, has_vertex_color=True, resolution=192)[0]


def reconstruct_hy(cutout_rgba, seed):
    """Hunyuan3D-2mini shape generation from an RGBA cutout. Shape only: no vertex colours."""
    try:
        out = S["hy"](image=cutout_rgba, num_inference_steps=HY_STEPS, octree_resolution=HY_RES, num_chunks=20000,
                      generator=torch.manual_seed(seed), output_type="trimesh")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        out = S["hy"](image=cutout_rgba, num_inference_steps=HY_STEPS, octree_resolution=256, num_chunks=8000,
                      generator=torch.manual_seed(seed), output_type="trimesh")
    mesh = out[0] if isinstance(out, (list, tuple)) else out
    return mesh


def reconstruct(cand, seed, backends, res=MC_RES):
    """Try backends in order; returns (mesh, backend_name). Any failure falls through to the next one."""
    last = None
    for b in backends:
        try:
            if b == "hy":
                return reconstruct_hy(cand["cutout"], seed), "hy"
            return reconstruct_tsr(cand["framed"], res), "tsr"
        except Exception as e:
            print(f"[{b} reconstruction failed] {e!r}")
            last = e
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise last or RuntimeError("no reconstruction backend loaded")


def proj_mask(xy, faces, size=160):
    mn, mx = xy.min(0), xy.max(0)
    s = (size - 6) / max(float((mx - mn).max()), 1e-9)
    px = ((xy[:, 0] - mn[0]) * s + 3).astype(np.int32)
    py = ((mx[1] - xy[:, 1]) * s + 3).astype(np.int32)
    img = np.zeros((size, size), np.uint8)
    tris = np.stack([px[faces], py[faces]], -1).astype(np.int32)
    for i in range(0, len(tris), 100000):
        cv2.fillPoly(img, tris[i:i + 100000], 255)
    return img > 0


def sil_iou_front(mesh, sil512):
    V = np.asarray(mesh.vertices)
    return iou(norm_mask(proj_mask(V[:, :2], np.asarray(mesh.faces))), norm_mask(sil512))


def align_mesh(mesh, sil512, backend="tsr"):
    """Find the orientation whose front projection best matches the sketch.
    TripoSR output is Z-up. Hunyuan3D is not guaranteed to be, so for it we also try identity and +90 deg about X.
    Candidates are ordered by prior likelihood; a later one only wins if it is clearly better (> +0.01 IoU)."""
    rx = lambda deg: trimesh.transformations.rotation_matrix(math.radians(deg), [1, 0, 0])
    pre = [rx(-90)] if backend == "tsr" else [np.eye(4), rx(-90), rx(90)]
    ref = norm_mask(sil512)
    best = (-1.0, 0, 0)
    for pi_, T0 in enumerate(pre):
        m0 = mesh.copy()
        m0.apply_transform(T0)
        V, F = np.asarray(m0.vertices), np.asarray(m0.faces)
        for k in range(4):
            c, sn = round(math.cos(k * math.pi / 2)), round(math.sin(k * math.pi / 2))
            x = V[:, 0] * c + V[:, 2] * sn
            i = iou(norm_mask(proj_mask(np.stack([x, V[:, 1]], 1), F)), ref)
            if i > best[0] + 0.01:
                best = (i, pi_, k)
    out = mesh.copy()
    out.apply_transform(pre[best[1]])
    if best[2]:
        out.apply_transform(trimesh.transformations.rotation_matrix(best[2] * math.pi / 2, [0, 1, 0]))
    return out, best[0]


# ============================================================ post-processing
def ensure_colors(mesh, default=(190, 178, 162)):
    try:
        vc = np.asarray(mesh.visual.vertex_colors)
        if len(vc) == len(mesh.vertices):
            return mesh
    except Exception:
        pass
    mesh.visual.vertex_colors = np.tile(np.array(list(default) + [255], np.uint8), (len(mesh.vertices), 1))
    return mesh


def decimate(mesh, n):
    if len(mesh.faces) <= n:
        return mesh
    try:
        try:
            out = mesh.simplify_quadric_decimation(face_count=n)
        except TypeError:
            out = mesh.simplify_quadric_decimation(n)
        if len(out.faces) == 0:
            return mesh
        vc = np.asarray(mesh.visual.vertex_colors)
        _, idx = cKDTree(mesh.vertices).query(out.vertices)
        out.visual.vertex_colors = vc[idx]
        return out
    except Exception as e:
        print("[decimate skipped]", e)
        return mesh


def finalize_mesh(mesh, z_mult=1.0):
    mesh = ensure_colors(mesh)
    parts = mesh.split(only_watertight=False)                      # connected-component isolation
    if len(parts) > 1:
        mesh = max(parts, key=lambda p: len(p.faces))
    mesh = decimate(mesh, FACE_BUDGET)
    try:
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except Exception as e:
        print("[merge skipped]", e)
    try:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=SMOOTH_ITERS)
    except Exception as e:
        print("[taubin skipped]", e)
    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
    except Exception as e:
        print("[repair skipped]", e)
    # Z anti-pancake: aspect-aware depth floor, then user feedback ("thinner" / "chunky")
    ext = mesh.extents
    width, height, depth = float(ext[0]), float(ext[1]), max(float(ext[2]), 1e-6)
    floor = 0.55 * min(width, height)
    f = min(1.8, floor / depth) if depth < floor else 1.0
    f = float(np.clip(f * z_mult, 0.5, 2.4))
    c = mesh.bounds.mean(0)
    mesh.apply_translation(-c)
    mesh.vertices[:, 2] *= f
    mesh.apply_translation(-mesh.bounds.mean(0))
    mesh.apply_scale(1.0 / float(max(mesh.extents)))
    return mesh, f


def health(mesh):
    d = dict(watertight=bool(mesh.is_watertight), winding_ok=bool(mesh.is_winding_consistent),
             extents=[round(float(x), 3) for x in mesh.extents], vertices=int(len(mesh.vertices)),
             faces=int(len(mesh.faces)), volume=None, euler=None)
    try:
        d["volume"] = round(abs(float(mesh.volume)), 4)
        d["euler"] = int(mesh.euler_number)
    except Exception:
        pass
    return d


# ============================================================ tier 0: CPU 2.5D inflation (always works, <1 s)
def tier0_mesh(sil512):
    m = norm_mask(sil512, 128, 0.92).astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dt = dt / max(float(dt.max()), 1e-6)
    h = 22.0 * np.sqrt(np.clip(1 - (1 - dt) ** 2, 0, 1))                # circular cross-section profile
    zs = np.arange(64) - 31.5
    vol = h[:, :, None] - np.abs(zs)[None, None, :]
    vol = ndi.gaussian_filter(vol, 1.0)
    vol = np.pad(vol, 1, constant_values=-5.0)                           # padding guarantees a closed surface
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0)
    verts = verts - 1.0
    xyz = np.stack([verts[:, 1], 128 - verts[:, 0], verts[:, 2] - 32.0], 1)   # col->x, row->-y (up), k->z
    mesh = trimesh.Trimesh(xyz, faces[:, ::-1], process=True)
    return ensure_colors(mesh)


# ============================================================ relighting presets (CPU, <0.3 s)
PRESETS = {
    "clay": dict(base=(184, 168, 150), spec=0.0, keep=0.55),
    "terracotta": dict(base=(197, 108, 74), spec=0.05, keep=0.50),
    "gold": dict(base=(232, 184, 74), spec=0.90, keep=0.35),
    "toy": dict(base=(94, 160, 232), spec=0.15, keep=0.40),
}
LIGHT = np.array([-0.4, 0.6, 0.7], np.float32)
LIGHT /= np.linalg.norm(LIGHT)


def apply_material(mesh, preset):
    p = PRESETS.get(preset)
    if not p:
        return mesh
    m = mesh.copy()
    V = np.asarray(m.vertices, np.float32)
    n = np.asarray(m.vertex_normals, np.float32)
    vc = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float32) / 255
    lum = vc @ np.array([0.299, 0.587, 0.114], np.float32)
    shade = np.clip(1 + p["keep"] * (lum - lum.mean()) * 2.0, 0.55, 1.35)
    diff = 0.62 + 0.38 * np.clip(n @ LIGHT, 0, 1)
    e = m.edges_unique
    A = sparse.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(len(V), len(V))).tocsr()
    deg = np.asarray(A.sum(1)).ravel() + 1e-6
    cav = (((A @ V) / deg[:, None] - V) * n).sum(1)                       # >0 in concavities
    ao = 1 - 0.35 * np.clip(cav / (np.percentile(np.abs(cav), 95) + 1e-9), 0, 1)
    Hh = LIGHT + np.array([0, 0, 1], np.float32)
    Hh /= np.linalg.norm(Hh)
    spec = np.clip(n @ Hh, 0, 1) ** 48 * p["spec"]
    col = np.array(p["base"], np.float32) / 255 * (shade * diff * ao)[:, None] + spec[:, None]
    rgba = np.concatenate([(col.clip(0, 1) * 255).astype(np.uint8), np.full((len(V), 1), 255, np.uint8)], 1)
    m.visual.vertex_colors = rgba
    return m


def _adj(mesh):
    V = np.asarray(mesh.vertices)
    e = mesh.edges_unique
    A = sparse.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V), len(V))).tocsr()
    return A, np.asarray(A.sum(1)).ravel() + 1e-6


def _set_colors(mesh, col):
    n = np.asarray(mesh.vertex_normals, np.float32)
    diff = 0.74 + 0.26 * np.clip(n @ LIGHT, 0, 1)
    col = (col * diff[:, None]).clip(0, 1)
    mesh.visual.vertex_colors = np.concatenate([(col * 255).astype(np.uint8), np.full((len(col), 1), 255, np.uint8)], 1)
    return mesh


def paint_from_cutout(mesh, cutout, back=None):
    """Hunyuan makes shape only. Project the winning 2D candidate's colours onto the mesh along the view axis.
    Surfaces that face away from the view axis (top, bottom, front, back) would smear, so they blend to the body colour."""
    arr = np.array(cutout.convert("RGBA"))
    msk = arr[..., 3] > 127
    y0, y1, x0, x1 = bbox(msk)
    rgb, msk = arr[y0:y1, x0:x1, :3], msk[y0:y1, x0:x1]
    body = np.median(rgb[msk].astype(np.float32), axis=0) / 255
    core = ndi.binary_erosion(msk, iterations=2)
    if core.sum() > 50:
        msk = core                                  # edge pixels are blended with the old background: colour comes from the inside
    _, (iy, ix) = ndi.distance_transform_edt(~msk, return_indices=True)
    rgb = rgb[iy, ix]
    h, w = msk.shape
    V = np.asarray(mesh.vertices)
    mn, mx = V[:, :2].min(0), V[:, :2].max(0)
    u = (V[:, 0] - mn[0]) / max(float(mx[0] - mn[0]), 1e-9)
    v = 1 - (V[:, 1] - mn[1]) / max(float(mx[1] - mn[1]), 1e-9)
    px = np.clip((u * (w - 1)).round().astype(int), 0, w - 1)
    py = np.clip((v * (h - 1)).round().astype(int), 0, h - 1)
    col = rgb[py, px].astype(np.float32) / 255
    n = np.asarray(mesh.vertex_normals, np.float32)
    wgt = np.clip((np.abs(n[:, 2]) - 0.25) / 0.45, 0, 1)[:, None]            # 1 where the surface faces the view axis
    bcol = None
    if back is not None:                                                      # real back colours from the Zero123++ back view, mirrored
        try:
            ba = np.array(back.convert("RGBA"))
            bm = ba[..., 3] > 127
            by0, by1, bx0, bx1 = bbox(bm)
            brgb, bm = ba[by0:by1, bx0:bx1, :3], bm[by0:by1, bx0:bx1]
            bbody = np.median(brgb[bm].astype(np.float32), axis=0) / 255
            _, (jy, jx) = ndi.distance_transform_edt(~bm, return_indices=True)
            brgb = brgb[jy, jx].astype(np.float32) / 255
            lum = lambda c: float(c @ np.array([0.299, 0.587, 0.114], np.float32))
            gain = float(np.clip(lum(body) / max(lum(bbody), 1e-3), 0.6, 1.6))                # same brightness as the front, hues untouched
            brgb = np.clip(brgb * gain, 0, 1)
            bh, bw = bm.shape
            bpx = np.clip(((1 - u) * (bw - 1)).round().astype(int), 0, bw - 1)
            bpy = np.clip((v * (bh - 1)).round().astype(int), 0, bh - 1)
            bcol = brgb[bpy, bpx]
        except Exception as e:
            print("[back colours skipped]", repr(e)[:200])
    if bcol is not None:
        wb = (np.clip((-n[:, 2] - 0.25) / 0.45, 0, 1) * 0.9)[:, None]
        wf = np.where(n[:, 2:3] > 0, wgt, 0.0)
        col = wf * col + wb * bcol + (1 - wf - wb) * body[None, :]
    else:
        col = wgt * col + (1 - wgt) * body[None, :]
        col[n[:, 2] < 0] *= 0.9
    A, deg = _adj(mesh)
    for _ in range(3):
        col = 0.5 * col + 0.5 * (A @ col) / deg[:, None]
    return _set_colors(mesh, col)


def recolor(mesh, rgb):
    """Guarantee the colour the user asked for: take the target hue and saturation, keep the existing shading and dark details."""
    vc = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255
    hsv = cv2.cvtColor(vc.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    th, ts, tv = cv2.cvtColor(np.array([[rgb]], np.float32) / 255, cv2.COLOR_RGB2HSV)[0, 0]
    v = hsv[:, 2]
    lit = v > 0.2
    vm = float(np.median(v[lit])) if lit.any() else 0.6
    v2 = np.clip(v / max(vm, 1e-3) * tv, 0, 1)
    s2 = np.where(lit, ts, ts * 0.3)
    out = np.stack([np.full_like(v, th), s2, v2], 1).astype(np.float32).reshape(-1, 1, 3)
    col = cv2.cvtColor(out, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    m = mesh.copy()
    m.visual.vertex_colors = np.concatenate([(col.clip(0, 1) * 255).astype(np.uint8), np.full((len(col), 1), 255, np.uint8)], 1)
    return m


def paint_flat(mesh, rgb):
    return _set_colors(mesh, np.tile(np.array(rgb, np.float32) / 255, (len(mesh.vertices), 1)))


# ============================================================ software turntable (no OpenGL needed)
VIEWS = [("Front", 0, 0), ("Side", 90, 0), ("Back", 180, 0), ("Isometric", 35, 25)]


def render_views(mesh, size=300, ss=2):
    proxy = decimate(mesh, 14000)
    V = np.asarray(proxy.vertices, np.float32)
    F = np.asarray(proxy.faces)
    fcol = np.asarray(proxy.visual.vertex_colors)[:, :3].astype(np.float32)[F].mean(1)
    r = float(np.linalg.norm(V, axis=1).max()) + 1e-9
    Sz = size * ss
    out = []
    for name, az, el in VIEWS:
        a, e = math.radians(az), math.radians(el)
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], np.float32)
        Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]], np.float32)
        P = V @ Ry.T @ Rx.T
        sc = 0.46 * Sz / r
        px = (P[:, 0] * sc + Sz / 2).astype(np.int32)
        py = (Sz / 2 - P[:, 1] * sc).astype(np.int32)
        tri = np.stack([px[F], py[F]], -1).astype(np.int32)
        a3, b3, c3 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        nrm = np.cross(b3 - a3, c3 - a3)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        shade = 0.38 + 0.62 * np.abs(nrm @ LIGHT)
        cols = (fcol * shade[:, None]).clip(0, 255).astype(np.uint8)
        order = np.argsort((a3[:, 2] + b3[:, 2] + c3[:, 2]))
        img = np.full((Sz, Sz, 3), 244, np.uint8)
        for i in order:
            cv2.fillConvexPoly(img, tri[i], (int(cols[i, 0]), int(cols[i, 1]), int(cols[i, 2])))
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out.append(dict(name=name, img=to_data_url(img, maxside=size)))
    return out


# ============================================================ exports
def export_all(mesh, jdir, tag="model"):
    glb = os.path.join(jdir, f"{tag}.glb")
    mesh.export(glb, file_type="glb")
    mesh.export(os.path.join(jdir, f"{tag}.obj"), file_type="obj")
    mesh.export(os.path.join(jdir, f"{tag}.stl"), file_type="stl")
    ok = False
    try:
        back = trimesh.load(glb, force="mesh")
        ok = len(back.faces) > 0 and abs(len(back.faces) - len(mesh.faces)) < 0.1 * len(mesh.faces) + 10
    except Exception as e:
        print("[reload check failed]", e)
    return ok


# ============================================================ the closed loop
def run_pipeline(img_bytes, feedback="", name="", mode="auto", force_cpu=False, progress=None, quality="fast", partial=None, multiview=False):
    P = progress or (lambda s: None)
    PART = partial or (lambda d: None)
    T = OrderedDict()
    t_all = time.time()
    job = uuid.uuid4().hex[:12]
    jdir = os.path.join(OUT_DIR, job)
    os.makedirs(jdir, exist_ok=True)

    rgb = load_rgb(img_bytes)
    kind = mode if mode in ("sketch", "photo") else detect_kind(rgb)
    Q = MODE_CFG.get(quality, MODE_CFG["fast"])
    fb = parse_feedback(f"{feedback} {name}")
    neural = (not force_cpu) and neural_ready()
    notes, attempts, cand_out, mv_out = [], [], [], []
    mv_back = None
    tier = "GPU neural pipeline" if neural else "CPU tier-0 (2.5D inflation)"
    used_backend = None
    if not force_cpu and not neural_ready():
        notes.append("GPU models not available, running CPU tier-0.")

    # ---- Stage 1
    P("Stage 1 · extracting silhouette and strokes" if kind == "sketch" else "Stage 1 · cutting out the object")
    t = time.time()
    if kind == "sketch":
        strokes, sil = extract_sketch(rgb)
        if not fb["color"]:
            sc = stroke_color(rgb, strokes)
            if sc:
                fb["color"] = sc
                fb["hits"].append(sc + " (from your drawing)")
        strokes512, sil512 = to_canvas(strokes, sil)
        control, paper = sketch_views(strokes512)
        photo_rgba = None
    else:
        small = rgb.copy()
        small.thumbnail((768, 768))
        if S["rembg"] is not None:
            photo_rgba = segment(small)
            pm = largest_open(photo_rgba[..., 3] > 127)
        else:
            _, pm = extract_sketch(small)
            photo_rgba = np.dstack([np.array(small), (pm * 255).astype(np.uint8)])
        strokes512 = np.zeros((CANVAS, CANVAS), bool)
        _, sil512 = to_canvas(pm, pm)
        control = paper = None
    T["Preprocess"] = round(time.time() - t, 2)

    # ---- Stage 2
    t = time.time()
    P("Stage 2 · labelling the object")
    user_label = (name or "").strip()
    S["label_conf"] = None
    S["label_top"] = []
    clip_label, clip_conf, clip_top = None, None, []
    if kind == "sketch" and neural:
        clip_label = identify_label(strokes512, sil512)
        clip_conf, clip_top = S["label_conf"], list(S["label_top"])
    label = user_label or clip_label or "object"
    src = "you typed it" if user_label else (S.get("label_source") or ("CLIP guess" if clip_label else "default"))
    if not user_label and label_is_low():
        notes.append(f"I guessed '{label}' but only with {S['label_conf']:.0%} confidence. If that is wrong, pick another option in the 'Identified as' card or type it in 'What is it?' and run again.")
    if fb["color"]:
        label = re.sub(r"\b" + fb["color"] + r"\b", "", label).strip() or "object"
    T["Label"] = round(time.time() - t, 2)
    identified = dict(label=label, source=src, caption="", view=None, clip_label=clip_label,
                      clip_conf=None if clip_conf is None else round(clip_conf, 3),
                      low=bool(not user_label and label_is_low()),
                      ai_note=S.get("ai_note"),
                      alts=[dict(label=l, p=round(p, 3)) for l, p in clip_top])
    PART(dict(identified=identified))                                    # the page shows "identified as" right away

    mesh, final_tier, best_iou = None, tier, None
    win_cutout = None
    used_backend = None
    prompt = ""
    if neural:
        try:
            # ---- Stage 3a
            t = time.time()
            adds = ", ".join(fb["add"])
            if kind == "sketch":
                subj = f"{fb['color']} {label}" if fb["color"] else label
                col_txt = f", painted {fb['color']}, {fb['color']} colored" if fb["color"] else ""
                cap_txt = ""
                view_txt = "front view"
                if S.get("sd_kind") == "xl":
                    prompt = (f"professional studio product photograph of a single centered {subj}, {adds + ', ' if adds else ''}"
                              f"real-life object{col_txt}, true-to-life proportions and materials, realistic surface texture, sharp focus, fine detail, "
                              f"soft even diffuse studio lighting, no harsh shadows, plain pure white seamless background, {view_txt}, "
                              f"entire object in frame, photorealistic, 8k{cap_txt}")
                    neg = ("flat, 2d, line art, sketch, drawing, illustration, cartoon, wireframe, text, watermark, cropped, cut off, "
                           "blurry, deformed, extra parts, harsh shadows, strong reflections, isometric, top-down, multiple objects, "
                           "busy background, low quality" + (", grey, monochrome" if fb["color"] else ""))
                else:
                    prompt = (f"a single centered {subj}, {adds + ', ' if adds else ''}product render{col_txt}, soft matte material, "
                              f"even diffuse studio lighting, no specular highlights, plain pure white background, {view_txt}, high quality{cap_txt}")
                    neg = ("flat, 2d, line art, sketch, wireframe, text, watermark, cropped, glossy, specular highlights, reflections, "
                           "harsh shadows, isometric, top-down, multiple objects, busy background, low quality"
                           + (", grey, monochrome" if fb["color"] else ""))
                P(f"Stage 3a · generating {Q['ncand']} shaded candidates ({'SDXL ' if S.get('sd_kind') == 'xl' else ''}ControlNet scribble)")
                seed = int(hashlib.sha1(img_bytes).hexdigest()[:8], 16) % (2 ** 31)
                imgs = gen_candidates(control, prompt, neg, fb["cn"], seed, Q["ncand"], Q["steps"])
                T["2D synthesis (SD+ControlNet)"] = round(time.time() - t, 2)
                t = time.time()
                P("Stage 3a · removing the background from the candidates")
                with ThreadPoolExecutor(len(imgs)) as ex:
                    rgbas = list(ex.map(segment, imgs))
                T["Segmentation"] = round(time.time() - t, 2)
            else:
                sq = np.array(Image.fromarray(photo_rgba[..., :3]))
                imgs = [Image.fromarray(sq)]
                rgbas = [photo_rgba]
                prompt = "(photo mode: no 2D synthesis)"

            # ---- Stage 4
            t = time.time()
            P("Stage 4 · scoring and ranking candidates")
            cands = []
            for img, rgba in zip(imgs, rgbas):
                arr = np.array(img.convert("RGB").resize((rgba.shape[1], rgba.shape[0])))
                mask = largest_open(rgba[..., 3] > 127)
                if kind == "sketch":
                    try:        # the render follows the sketch, so anything far outside its silhouette (shadow, floor, leftovers) is background
                        hh, ww = mask.shape
                        bound = cv2.resize(sil512.astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST)
                        kk = max(15, int(0.06 * hh)) | 1
                        bound = cv2.dilate(bound, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))) > 0
                        if (mask & bound).sum() >= 0.85 * mask.sum():
                            mask = largest_open(mask & bound)
                    except Exception as e:
                        print("[sketch bound skipped]", repr(e)[:200])
                    mask = open_gaps(mask, arr)
                ok = mask.mean() > 0.01
                if kind == "sketch" and ok:
                    arr512 = cv2.resize(arr, (CANVAS, CANVAS))
                    mask512 = cv2.resize(mask.astype(np.uint8), (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST) > 0
                    total, parts = score_candidate(strokes512, sil512, arr512, mask512, label)
                else:
                    total, parts = (1.0 if ok else 0.0), {}
                framed = frame_fg(arr, mask) if ok else None
                cutout = Image.fromarray(np.dstack([edge_colors(arr, mask), mask.astype(np.uint8) * 255]), "RGBA") if ok else None
                cands.append(dict(img=img, framed=framed, cutout=cutout, score=total, parts=parts, ok=ok))
            order = sorted([i for i, c in enumerate(cands) if c["ok"]], key=lambda i: -cands[i]["score"])
            T["Scoring"] = round(time.time() - t, 2)
            if not order:
                raise RuntimeError("no usable candidate (segmentation found nothing)")
            PART(dict(identified=identified, candidates=[
                dict(img=to_data_url(c["framed"] if c["framed"] is not None else c["img"], maxside=320), score=round(c["score"], 3),
                     parts={k: round(v, 3) for k, v in c["parts"].items()}, winner=(i == order[0])) for i, c in enumerate(cands)]))
            # ---- Stage 3b + 8 (closed loop)
            t = time.time()
            best = None
            chain = backend_chain(Q["backend"])
            seed3d = int(hashlib.sha1(img_bytes).hexdigest()[8:16], 16) % (2 ** 31)
            if multiview and MV_ON:
                ci0 = order[0]
                try:
                    P("Stage 3b · multi-view: Zero123++ is drawing the side and back views (first run downloads the models)")
                    tm = time.time()
                    tiles = mv_generate(cands[ci0]["cutout"], seed3d)
                    T["Multi-view images (Zero123++)"] = round(time.time() - tm, 2)
                    views_mv, mv_out = mv_views(cands[ci0]["cutout"], tiles)
                    mv_back = views_mv.get("back")
                    PART(dict(mv_views=mv_out))
                    P("Stage 3b · multi-view: Hunyuan3D-2mv is building the shape from front, left and back")
                    tm = time.time()
                    raw = reconstruct_hymv(views_mv, seed3d)
                    T["Multi-view 3D (Hunyuan3D-2mv)"] = round(time.time() - tm, 2)
                    aligned, i3 = align_mesh(raw, sil512, "hy")
                    attempts.append(dict(candidate=ci0 + 1, iou=round(i3, 3), backend="hymv"))
                    P(f"Loop 2 · multi-view 3D silhouette IoU = {i3:.2f}")
                    best = (i3, aligned, ci0, "hymv")
                    if i3 < IOU_RETRY:
                        notes.append(f"Multi-view mesh matched your sketch only {i3:.2f}: also trying single-view TripoSR and keeping the better one.")
                except Exception as e:
                    traceback.print_exc()
                    print("[multi-view error]", repr(e)[:600], flush=True)
                    notes.append(f"Multi-view failed ({type(e).__name__}: {str(e)[:200]}). Used single-view TripoSR instead.")
                    best = None
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            for rank, ci in enumerate(order[:2] if (best is None or best[0] < IOU_RETRY) else []):
                P(f"Stage 3b · {'Hunyuan3D' if chain[0] == 'hy' else 'TripoSR'} reconstruction from candidate {ci + 1}")
                raw, used = reconstruct(cands[ci], seed3d, chain, Q.get("res", MC_RES))
                aligned, i3 = align_mesh(raw, sil512, used)
                attempts.append(dict(candidate=ci + 1, iou=round(i3, 3), backend=used))
                P(f"Loop 2 · 3D silhouette IoU = {i3:.2f}")
                if best is None or i3 > best[0] + (0.15 if best[3] == "hymv" else 0.0):
                    best = (i3, aligned, ci, used)
                if i3 >= IOU_RETRY:
                    break
                if rank == 0 and len(order) > 1:
                    P(f"Loop 2 · below {IOU_RETRY:.2f}, retrying with the next candidate")
            T["3D reconstruction + verification"] = round(time.time() - t, 2)
            best_iou, mesh, win, used_backend = best
            if used_backend == "hymv":
                notes.append("Multi-view: the shape was built by Hunyuan3D-2mv from your image plus Zero123++ left and back views. "
                             "Front colours come from the photoreal render, back colours from the Zero123++ back view.")
            win_cutout = cands[win]["cutout"]
            order_winner = win
            if best_iou < IOU_FALLBACK:
                notes.append(f"Neural mesh IoU {best_iou:.2f} < {IOU_FALLBACK:.2f}: fell back to tier-0 mesh.")
                mesh, final_tier, used_backend = None, "CPU tier-0 (fidelity gate fallback)", None
            else:
                if best_iou < IOU_RETRY:
                    notes.append("Your sketch is loose, so the model follows it more freely. Type what the object is in 'What is it?' to guide it.")
                for i, c in enumerate(cands):
                    cand_out.append(dict(img=to_data_url(c["framed"] if c["framed"] is not None else c["img"], maxside=320),
                                         score=round(c["score"], 3),
                                         parts={k: round(v, 3) for k, v in c["parts"].items()},
                                         winner=(i == order_winner)))
        except Exception as e:
            traceback.print_exc()
            print("[neural pipeline error]", repr(e)[:600], flush=True)
            notes.append(f"Neural pipeline failed ({type(e).__name__}): {str(e)[:260]}. Switched to CPU tier-0.")
            final_tier = "CPU tier-0 (error fallback)"
            mesh, used_backend = None, None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if mesh is None:
        t = time.time()
        P("Tier 0 · inflating silhouette on CPU")
        mesh = tier0_mesh(sil512)
        T["Tier-0 mesh"] = round(time.time() - t, 2)
        if final_tier == tier and neural:
            final_tier = "CPU tier-0"
        elif not neural:
            final_tier = tier

    # ---- post-process, verify, export
    t = time.time()
    P("Post-processing · components, decimate, Taubin smoothing, normals, depth")
    mesh, zf = finalize_mesh(mesh, fb["z"])
    shape_only = used_backend in ("hy", "hymv") and mesh is not None
    if shape_only and win_cutout is not None:
        try:
            mesh = paint_from_cutout(mesh, win_cutout, mv_back if used_backend == "hymv" else None)
        except Exception as e:
            print("[paint skipped]", e)
            fb["preset"] = fb["preset"] or "clay"
    elif shape_only and not fb["preset"]:
        fb["preset"] = "clay"
    elif used_backend is None and fb["color"]:
        mesh = paint_flat(mesh, COLOR_RGB[fb["color"]])
    if fb["color"] and not fb["preset"]:
        mesh = recolor(mesh, COLOR_RGB[fb["color"]])
    final_iou = sil_iou_front(mesh, sil512)
    mesh.export(os.path.join(jdir, "orig.ply"), file_type="ply")
    shown = apply_material(mesh, fb["preset"]) if fb["preset"] else mesh
    T["Post-process"] = round(time.time() - t, 2)

    t = time.time()
    P("Exporting .glb / .obj / .stl and verifying reload")
    reload_ok = export_all(shown, jdir)
    views = render_views(shown)
    T["Export + 4-view render"] = round(time.time() - t, 2)
    P("Done")

    rep = health(shown)
    rep.update(backend={"hy": "Hunyuan3D-2mini", "hymv": "Hunyuan3D-2mv (multi-view)", "tsr": "TripoSR"}.get(used_backend, "tier-0"), sil_iou=round(final_iou, 3), neural_iou=None if best_iou is None else round(best_iou, 3),
               z_boost=round(zf, 2), reload_ok=bool(reload_ok), attempts=attempts,
               iou_retry=IOU_RETRY, iou_fallback=IOU_FALLBACK)
    return dict(job=job, kind=kind, label=label, identified=identified, tier=final_tier, notes=notes, prompt=prompt,
                feedback=dict(hits=fb["hits"], z=round(fb["z"], 2), cn=fb["cn"], preset=fb["preset"]),
                glb=f"/files/{job}/model.glb", preset=fb["preset"] or "original",
                downloads=dict(glb=f"/files/{job}/model.glb", obj=f"/files/{job}/model.obj", stl=f"/files/{job}/model.stl"),
                candidates=cand_out, mv_views=mv_out, views=views, report=rep, timings=T, total=round(time.time() - t_all, 2))



# ============================================================ separate feature: 4 photos -> 3D (no sketch, no 2D synthesis)
PHOTO_SLOTS = ["front", "left", "back", "right"]
PHOTO_NAMES = dict(front="Front photo", left="Left side photo", back="Back photo", right="Right side photo")


def photo_cutout(raw, slot):
    im = load_rgb(raw)
    im.thumbnail((768, 768))
    rgba = segment(im)
    m = open_gaps(largest_open(rgba[..., 3] > 127), np.array(im.convert("RGB")))
    if m.mean() < 0.01:
        raise ValueError(f"Could not find the object in the {slot} photo. Use a plain background and keep the whole object in frame.")
    rgba = rgba.copy()
    rgba[..., 3] = m.astype(np.uint8) * 255
    return rgba


def view_check(mesh, slot, mask512):
    """Rough self-check for one photo: silhouette of the finished mesh seen from that side vs the photo's cut-out."""
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    ks = {"front": (0,), "back": (2,), "left": (1, 3), "right": (1, 3)}[slot]
    ref = norm_mask(mask512)
    best = 0.0
    for k in ks:
        c, sn = round(math.cos(k * math.pi / 2)), round(math.sin(k * math.pi / 2))
        x = V[:, 0] * c + V[:, 2] * sn
        pm = norm_mask(proj_mask(np.stack([x, V[:, 1]], 1), F))
        best = max(best, iou(pm, ref), iou(pm[:, ::-1], ref) if slot != "front" else 0.0)
    return round(best, 3)


def run_photos(raws, name="", quality="fast", progress=None, partial=None):
    """raws: dict slot -> image bytes (front + at least one more). Builds one mesh from real photos."""
    P = progress or (lambda s: None)
    PART = partial or (lambda d: None)
    T = OrderedDict()
    t_all = time.time()
    if not neural_ready():
        raise ValueError("The GPU models are not loaded, so the photo mode cannot run.")
    if "front" not in raws or len(raws) < 2:
        raise ValueError("Add a front photo and at least one more angle (left, back or right).")
    job = uuid.uuid4().hex[:12]
    jdir = os.path.join(OUT_DIR, job)
    os.makedirs(jdir, exist_ok=True)
    fb = parse_feedback(name)
    notes, attempts = [], []
    Q = MODE_CFG.get(quality, MODE_CFG["fast"])

    t = time.time()
    P(f"Stage 1 · cutting the object out of {len(raws)} photos")
    slots = [k for k in PHOTO_SLOTS if k in raws]
    rgbas = [photo_cutout(raws[k], k) for k in slots]
    masks = {k: a[..., 3] > 127 for k, a in zip(slots, rgbas)}
    T["Cut-outs"] = round(time.time() - t, 2)
    label = (name or "").strip() or "object"
    PART(dict(identified=dict(label=label, source="your photos", caption="", view=None, clip_label=None, clip_conf=None, alts=[])))

    norm = norm_views(rgbas)
    if norm is None:
        raise ValueError("One of the photos has an almost empty cut-out. Retake it with a plain background.")
    views = dict(zip(slots, norm))
    bg = lambda im: Image.alpha_composite(Image.new("RGBA", im.size, (240, 240, 240, 255)), im.convert("RGBA")).convert("RGB")
    mv_out = [dict(img=to_data_url(bg(views[k]), maxside=240), name=PHOTO_NAMES[k], used=True) for k in slots]
    PART(dict(mv_views=mv_out))
    front_cut = views["front"]
    _, sil512 = to_canvas(masks["front"], masks["front"])
    seed = int(hashlib.sha1(b"".join(raws[k] for k in slots)).hexdigest()[:8], 16) % (2 ** 31)

    t = time.time()
    mesh, used_backend, best_iou = None, None, None
    try:
        P(f"Stage 3b · Hunyuan3D-2mv is building the shape from {len(slots)} photos (first run downloads the model)")
        raw = reconstruct_hymv(views, seed)
        mesh, best_iou = align_mesh(raw, sil512, "hy")
        used_backend = "hymv"
        attempts.append(dict(candidate=1, iou=round(best_iou, 3), backend="hymv"))
        P(f"Loop 2 · front photo vs 3D silhouette = {best_iou:.2f}")
    except Exception as e:
        traceback.print_exc()
        notes.append(f"Multi-photo reconstruction failed ({type(e).__name__}: {str(e)[:200]}). Used the front photo only, with TripoSR.")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if mesh is None:
        P("Stage 3b · TripoSR is building from the front photo")
        a = np.array(front_cut)
        framed = frame_fg(a[..., :3], a[..., 3] > 127)
        raw = reconstruct_tsr(framed, Q.get("res", MC_RES))
        mesh, best_iou = align_mesh(raw, sil512, "tsr")
        used_backend = "tsr"
        attempts.append(dict(candidate=1, iou=round(best_iou, 3), backend="tsr"))
    T["3D reconstruction"] = round(time.time() - t, 2)

    t = time.time()
    P("Post-processing · components, decimate, Taubin smoothing, normals, depth")
    mesh, zf = finalize_mesh(mesh, fb["z"])
    try:
        mesh = paint_from_cutout(mesh, front_cut)
    except Exception as e:
        print("[paint skipped]", e)
    if fb["color"] and not fb["preset"]:
        mesh = recolor(mesh, COLOR_RGB[fb["color"]])
    final_iou = sil_iou_front(mesh, sil512)
    checks = {k: view_check(mesh, k, masks[k]) for k in slots}
    mesh.export(os.path.join(jdir, "orig.ply"), file_type="ply")
    shown = apply_material(mesh, fb["preset"]) if fb["preset"] else mesh
    T["Post-process"] = round(time.time() - t, 2)

    t = time.time()
    P("Exporting .glb / .obj / .stl and verifying reload")
    reload_ok = export_all(shown, jdir)
    vws = render_views(shown)
    T["Export + 4-view render"] = round(time.time() - t, 2)
    P("Done")
    notes.append("Per-photo check (how closely the finished model's outline matches each photo, rough): "
                 + ", ".join(f"{PHOTO_NAMES[k].split(' photo')[0].lower()} {int(v * 100)}%" for k, v in checks.items())
                 + ". A low number on one photo usually means that photo is at a different angle or distance than the others.")
    if len(slots) < 4:
        notes.append(f"Only {len(slots)} photos were given. Sides you did not photograph are guessed by the model; all 4 angles work best.")
    rep = health(shown)
    rep.update(backend={"hymv": "Hunyuan3D-2mv (photos)", "tsr": "TripoSR (front photo only)"}.get(used_backend, "?"), sil_iou=round(final_iou, 3),
               neural_iou=round(best_iou, 3), z_boost=round(zf, 2), reload_ok=bool(reload_ok), attempts=attempts,
               iou_retry=IOU_RETRY, iou_fallback=IOU_FALLBACK, per_view=checks)
    return dict(job=job, kind="photos", label=label, identified=None, tier="Photos to 3D", notes=notes, prompt="",
                feedback=dict(hits=fb["hits"], z=round(fb["z"], 2), cn=fb["cn"], preset=fb["preset"]),
                glb=f"/files/{job}/model.glb", preset=fb["preset"] or "original",
                downloads=dict(glb=f"/files/{job}/model.glb", obj=f"/files/{job}/model.obj", stl=f"/files/{job}/model.stl"),
                candidates=[], mv_views=mv_out, views=vws, report=rep, timings=T, total=round(time.time() - t_all, 2))


def relight(job, preset):
    if not re.fullmatch(r"[a-f0-9]{12}", job or ""):
        raise ValueError("bad job id")
    jdir = os.path.join(OUT_DIR, job)
    ply = os.path.join(jdir, "orig.ply")
    if not os.path.exists(ply):
        raise ValueError("job not found (server restarted?)")
    m = trimesh.load(ply, force="mesh", process=False)
    shown = apply_material(m, preset) if preset in PRESETS else m
    tag = f"model_{preset if preset in PRESETS else 'original'}"
    export_all(shown, jdir, tag)
    return dict(glb=f"/files/{job}/{tag}.glb", views=render_views(shown),
                downloads=dict(glb=f"/files/{job}/{tag}.glb", obj=f"/files/{job}/{tag}.obj", stl=f"/files/{job}/{tag}.stl"))


# ============================================================ web app
from fastapi import Body, FastAPI  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402

app = FastAPI(title="dribble3d")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/health")
def health_ep():
    return dict(device=S["device"], neural_ready=neural_ready(), errors=S["err"], loaded=S["loaded"], mv=dict(on=MV_ON, default=MV_DEFAULT), mem=_mem())


@app.get("/api/progress/{rid}")
def progress_ep(rid: str):
    return dict(steps=PROGRESS.get(rid, []), partial=PARTIAL.get(rid, {}))


RESULTS: dict = {}


@app.post("/api/generate")
def generate_ep(payload: dict = Body(...)):
    """Starts the run in a background thread and answers at once. A tunnel closes any request that stays open for about 100 s,
    so the page polls /api/result/<rid> instead of waiting on this call."""
    rid = str(payload.get("rid", ""))[:32] or uuid.uuid4().hex[:12]
    PROGRESS[rid] = []
    PARTIAL[rid] = {}
    RESULTS.pop(rid, None)
    t0 = time.time()
    if len(PROGRESS) > 200:
        PROGRESS.pop(next(iter(PROGRESS)))
    while len(PARTIAL) > 40:
        PARTIAL.pop(next(iter(PARTIAL)))
    while len(RESULTS) > 20:
        RESULTS.pop(next(iter(RESULTS)))

    def PT(d):
        PARTIAL.setdefault(rid, {}).update(d)

    def P(msg):
        PROGRESS.setdefault(rid, []).append(dict(t=round(time.time() - t0, 1), msg=msg))

    def work():
        try:
            data = payload["image"].split(",", 1)[1]
            raw = base64.b64decode(data)
            key = hashlib.sha1(raw + repr((payload.get("feedback"), payload.get("name"), payload.get("mode"), payload.get("cpu"), payload.get("quality"), payload.get("multiview", MV_ON))).encode()).hexdigest()
            if key in _CACHE:
                P("Served from cache")
                RESULTS[rid] = dict(status="done", result=_CACHE[key])
                return
            P("Queued")
            with _LOCK:
                res = run_pipeline(raw, payload.get("feedback", ""), payload.get("name", ""), payload.get("mode", "auto"),
                                   bool(payload.get("cpu")), P, quality=str(payload.get("quality") or "fast"), partial=PT,
                                   multiview=bool(payload.get("multiview", MV_ON)))
            _CACHE[key] = res
            while len(_CACHE) > KEEP_JOBS:
                _CACHE.popitem(last=False)
            RESULTS[rid] = dict(status="done", result=res)
            _prune_outputs()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(_mem(), flush=True)
        except ValueError as e:
            RESULTS[rid] = dict(status="error", error=str(e))
        except Exception as e:
            print("[generate error]", repr(e))
            RESULTS[rid] = dict(status="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=work, daemon=True).start()
    return dict(rid=rid, status="started")


@app.post("/api/photos3d")
def photos_ep(payload: dict = Body(...)):
    """Separate feature: 2-4 photos of one object from different sides -> one 3D model. Same background-job pattern as /api/generate."""
    rid = str(payload.get("rid", ""))[:32] or uuid.uuid4().hex[:12]
    PROGRESS[rid] = []
    PARTIAL[rid] = {}
    RESULTS.pop(rid, None)
    t0 = time.time()

    def PT(d):
        PARTIAL.setdefault(rid, {}).update(d)

    def P(msg):
        PROGRESS.setdefault(rid, []).append(dict(t=round(time.time() - t0, 1), msg=msg))

    def work():
        try:
            raws = {}
            for slot, du in zip(PHOTO_SLOTS, (payload.get("images") or [])[:4]):
                if du:
                    raws[slot] = base64.b64decode(du.split(",", 1)[1])
            P("Queued")
            with _LOCK:
                res = run_photos(raws, payload.get("name", ""), str(payload.get("quality") or "fast"), P, PT)
            RESULTS[rid] = dict(status="done", result=res)
            _prune_outputs()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(_mem(), flush=True)
        except ValueError as e:
            RESULTS[rid] = dict(status="error", error=str(e))
        except Exception as e:
            traceback.print_exc()
            RESULTS[rid] = dict(status="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=work, daemon=True).start()
    return dict(rid=rid, status="started")


@app.get("/api/result/{rid}")
def result_ep(rid: str):
    return RESULTS.get(rid) or dict(status="running")


@app.post("/api/relight")
def relight_ep(payload: dict = Body(...)):
    try:
        return relight(str(payload.get("job", "")), str(payload.get("preset", "original")))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


_MEDIA = {".glb": "model/gltf-binary", ".obj": "text/plain", ".stl": "model/stl", ".ply": "application/octet-stream"}


@app.get("/files/{job}/{fname}")
def files_ep(job: str, fname: str):
    if not re.fullmatch(r"[a-f0-9]{12}", job) or not re.fullmatch(r"model[\w]*\.(glb|obj|stl)", fname):
        return JSONResponse({"error": "not found"}, status_code=404)
    p = os.path.join(OUT_DIR, job, fname)
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type=_MEDIA[os.path.splitext(fname)[1]], filename=fname if not fname.endswith(".glb") else None)


# ============================================================ UI (single page: white light mode, Lexend, light/dark/auto)
HTML = r"""<!doctype html>
<html lang="en" data-theme="light" data-pref="auto"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>dribble3d</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@300..700&display=swap" rel="stylesheet">
<script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<script>
(function(){var p='auto';try{p=localStorage.getItem('d3t')||'auto'}catch(e){}
var m=p==='auto'?(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'):p;
var h=document.documentElement;h.dataset.theme=m;h.dataset.pref=p})();
</script>
<style>
/* ---------- tokens ----------
   Type: Lexend everywhere.
   Elevation scale (same light direction everywhere):
     0 recessed  fields, segmented tracks        inset only
     1 control   pills, chips, small buttons     e1
     2 surface   cards, panels                   e2
     3 raised    hover on controls, hero stat    e3
   Tiles inside a card are flat (tint + hairline, no shadow) so shadows never nest. */
:root{--sans:'Lexend',ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;
--serif:var(--sans);
--ease:cubic-bezier(.22,.8,.24,1);--spring:cubic-bezier(.34,1.35,.5,1);
color-scheme:light;
--page:#fff;--card:#fff;--tile:#f7f7f9;--recess:#f2f2f5;--field:#fff;--thumb:#fff;
--ink:#15151a;--muted:#5b5b66;--soft:#73737e;--hair:#e3e3e8;--line:rgba(21,21,26,.09);
--accent:#d97757;--accent-ink:#b5502c;--accent-soft:rgba(217,119,87,.13);
--btn:#15151a;--btn-hover:#2b2b33;--btn-text:#fff;--good:#2a8a57;--warn:#c23b2b;--warn-soft:rgba(194,59,43,.09);
--glow:transparent;--shine:inset 0 1px 0 rgba(255,255,255,0);
--e1:0 1px 2px rgba(21,21,26,.07),0 0 0 .5px rgba(21,21,26,.03);
--e2:0 1px 2px rgba(21,21,26,.04),0 3px 8px -2px rgba(21,21,26,.06),0 14px 28px -10px rgba(21,21,26,.10);
--e3:0 2px 4px rgba(21,21,26,.05),0 10px 20px -4px rgba(21,21,26,.09),0 26px 44px -12px rgba(21,21,26,.16);
--well:inset 0 1px 2px rgba(21,21,26,.07);
--btn-sh:inset 0 1px 0 rgba(255,255,255,.14),0 1px 2px rgba(21,21,26,.22),0 8px 18px -8px rgba(21,21,26,.45);
--well-a:#f2f2f5;--well-b:#fbfbfd;--view-a:#fff;--view-b:#f6f6f9;--view-c:#ebebf0;
--cube-edge:rgba(21,21,26,.8);--cube-a:rgba(255,255,255,.85);--cube-b:rgba(217,119,87,.14)}
html[data-theme=dark]{color-scheme:dark;--page:#0e0e0f;--card:#171718;--tile:#1e1e20;--recess:#111112;--field:#111112;--thumb:#2a2a2d;
--ink:#f3f3ef;--muted:#a8a8a2;--soft:#7a7a74;--hair:#2c2c2f;--line:rgba(255,255,255,.09);
--accent:#e08a6a;--accent-ink:#ee9f82;--accent-soft:rgba(224,138,106,.20);
--btn:#f3f3ef;--btn-hover:#fff;--btn-text:#111;--good:#5cc28a;--warn:#ee7d70;--warn-soft:rgba(238,125,112,.12);
--glow:rgba(217,119,87,.14);--shine:inset 0 1px 0 rgba(255,255,255,.05);
--e1:0 1px 2px rgba(0,0,0,.5);
--e2:0 1px 2px rgba(0,0,0,.4),0 10px 26px -8px rgba(0,0,0,.6);
--e3:0 2px 4px rgba(0,0,0,.45),0 18px 40px -10px rgba(0,0,0,.7);
--well:inset 0 1px 3px rgba(0,0,0,.5);
--btn-sh:inset 0 1px 0 rgba(255,255,255,.5),0 1px 2px rgba(0,0,0,.4),0 8px 18px -8px rgba(0,0,0,.6);
--well-a:#161617;--well-b:#222224;--view-a:#2a2a2c;--view-b:#1b1b1c;--view-c:#121213;
--cube-edge:rgba(243,243,239,.85);--cube-a:rgba(255,255,255,.06);--cube-b:rgba(224,138,106,.18)}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scrollbar-gutter:stable;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--sans);font-size:13.5px;line-height:1.5;font-weight:400;
-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;text-rendering:optimizeLegibility}
body::before{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;background:radial-gradient(900px 420px at 88% -8%,var(--glow),transparent 60%)}
button,input,select{font-family:inherit}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.wrap{max-width:min(1500px,94vw);margin:0 auto;padding:0 20px 64px}
header{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:26px 0 20px}
.brand{display:flex;align-items:center;gap:14px}
.mark{width:42px;height:42px;border-radius:13px;display:grid;place-items:center;background:var(--accent);color:#fff;box-shadow:0 1px 2px rgba(181,80,44,.3),0 8px 18px -8px var(--accent)}
.brand h1{margin:0;font-family:var(--serif);font-size:28px;font-weight:500;letter-spacing:-.02em;line-height:1.05}.brand h1 b{color:var(--accent);font-weight:500}
.brand p{margin:5px 0 0;color:var(--muted);font-size:13.5px}
/* theme switch with a sliding thumb */
.switch{position:relative;display:inline-flex;gap:2px;padding:4px;border-radius:14px;background:var(--recess);border:1px solid var(--line);box-shadow:var(--well)}
.switch::before{content:"";position:absolute;left:4px;top:4px;width:34px;height:32px;border-radius:10px;background:var(--btn);box-shadow:var(--btn-sh);transform:translateX(calc(var(--pos,2)*36px));transition:transform .5s var(--spring)}
html[data-pref=light] .switch{--pos:0}html[data-pref=dark] .switch{--pos:1}html[data-pref=auto] .switch{--pos:2}
.switch button{position:relative;z-index:1;width:34px;height:32px;display:grid;place-items:center;border:0;border-radius:10px;background:transparent;color:var(--muted);cursor:pointer;transition:color .3s var(--ease)}
.switch button:hover{color:var(--ink)}
html[data-pref=light] .switch [data-t=light],html[data-pref=dark] .switch [data-t=dark],html[data-pref=auto] .switch [data-t=auto]{color:var(--btn-text)}
/* layout: the left column scrolls on its own when it is taller than the window (it used to get stuck) */
.grid{display:grid;grid-template-columns:minmax(340px,5fr) 7fr;gap:24px;align-items:start}
.left{position:sticky;top:16px;max-height:calc(100vh - 32px);max-height:calc(100dvh - 32px);overflow-y:auto;overflow-x:hidden;
padding:8px 18px 32px;margin:-8px -18px 0;scrollbar-width:thin;scrollbar-color:var(--hair) transparent}
.left::-webkit-scrollbar{width:8px}.left::-webkit-scrollbar-thumb{background:var(--hair);border-radius:8px}
@media(max-width:900px){.grid{grid-template-columns:1fr}.left{position:static;max-height:none;overflow:visible;padding:0;margin:0}}
/* surfaces (level 2) */
.card,.think-head,.trace,.idle,.stat{background:var(--card);border:1px solid var(--line);box-shadow:var(--e2),var(--shine)}
.card{border-radius:20px;padding:20px;margin-bottom:16px}
.lab{display:block;margin:0 0 10px;font-size:12.5px;font-weight:500;color:var(--muted)}
/* one orchestrated page-load: left cards settle in, one after another */
.reveal{animation:fadeUp .6s var(--ease) backwards;animation-delay:calc(var(--k,0)*90ms)}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.enter{animation:fadeUp .5s var(--ease) backwards}
.leave{opacity:0;transform:translateY(-6px);transition:opacity .22s ease,transform .22s ease;pointer-events:none}
#res.enter,#top.enter{animation:none}
#res.enter>*,#top.enter>*{animation:fadeUp .7s var(--ease) backwards;animation-delay:calc(120ms + var(--k,0)*90ms)}
#res.enter>.stats{animation:none}
#res>:nth-child(2),#top>:nth-child(2){--k:1}#res>:nth-child(3){--k:2}#res>:nth-child(4){--k:3}#res>:nth-child(5){--k:4}#res>:nth-child(6){--k:5}#res>:nth-child(7){--k:6}
#pad{width:100%;aspect-ratio:1;border-radius:14px;background:#fff;touch-action:none;cursor:crosshair;box-shadow:var(--well),0 0 0 1px var(--hair);display:block}
#pad.scroll{touch-action:pan-y pinch-zoom}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px}
.pill{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:999px;padding:7px 14px;font-size:13px;font-weight:450;cursor:pointer;box-shadow:var(--e1);
transition:transform .25s var(--ease),box-shadow .25s var(--ease),color .2s,background-color .25s,border-color .25s}
.pill:hover{transform:translateY(-1px);box-shadow:var(--e2);color:var(--accent-ink)}
.pill:active{transform:scale(.97);box-shadow:var(--e1)}
.pill.on{background:var(--btn);color:var(--btn-text);border-color:transparent;box-shadow:var(--btn-sh)}
.pill.on:hover{color:var(--btn-text)}
.pill.chip{color:var(--muted);padding:6px 13px;font-weight:400}
.scrollpill{display:none}@media(pointer:coarse){.scrollpill{display:inline-flex}}
/* segmented controls with a sliding thumb */
.seg{position:relative;display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:5px;border-radius:16px;background:var(--recess);border:1px solid var(--line);box-shadow:var(--well)}
.seg::before{content:"";position:absolute;top:5px;bottom:5px;left:5px;width:calc((100% - 16px)/2);border-radius:12px;background:var(--thumb);box-shadow:var(--e1),var(--shine);
transform:translateX(calc(var(--pos,0)*(100% + 6px)));transition:transform .5s var(--spring)}
.seg button{position:relative;z-index:1;border:0;border-radius:12px;padding:10px 12px;background:transparent;color:var(--muted);font-size:13.5px;font-weight:500;cursor:pointer;text-align:left;transition:color .3s var(--ease)}
.seg button small{display:block;font-size:11.5px;font-weight:400;margin-top:2px;color:var(--soft)}
.seg button.on{color:var(--ink)}
.hint{margin:8px 2px 0;font-size:12px;line-height:1.45;color:var(--soft)}.hint.bad{color:var(--warn)}
button.go{width:100%;margin-top:18px;min-height:50px;border:0;border-radius:999px;background:var(--btn);color:var(--btn-text);font-size:15px;font-weight:500;cursor:pointer;box-shadow:var(--btn-sh);
transition:transform .3s var(--ease),background-color .25s,opacity .25s,box-shadow .3s var(--ease)}
button.go:hover{background:var(--btn-hover);transform:translateY(-2px)}button.go:active{transform:scale(.985)}button.go:disabled{opacity:.55;cursor:wait;transform:none}
input[type=text],select{width:100%;background:var(--field);border:1px solid var(--hair);color:var(--ink);border-radius:12px;padding:10px 12px;font-size:14px;outline:none;box-shadow:var(--well);transition:border-color .25s,box-shadow .25s}
input[type=text]::placeholder{color:var(--soft)}
input[type=text]:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
input[type=range]{accent-color:var(--accent)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.chk{display:flex;gap:8px;align-items:center;font-size:13px;color:var(--muted);margin-top:14px}
.chk input{accent-color:var(--accent)}
.stage{min-height:560px}
/* idle */
.idle{position:relative;display:flex;flex-direction:column;gap:20px;padding:38px 34px 34px;border-radius:24px;overflow:hidden;margin-bottom:16px}
.idle .art{display:flex;align-items:center;gap:26px}
.idle h3{margin:0 0 8px;font-family:var(--serif);font-size:25px;font-weight:500;letter-spacing:-.02em;line-height:1.2}
.idle p{margin:0;max-width:400px;font-size:14.5px;line-height:1.6;color:var(--muted)}
.idle ol{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:520px}
.idle li{display:flex;gap:10px;align-items:baseline;font-size:13.5px;color:var(--muted);padding:10px 14px;border-radius:12px;background:var(--tile);border:1px solid var(--line)}
.idle li b{font-weight:500;color:var(--accent-ink);font-size:12px;font-variant-numeric:tabular-nums}
/* cube */
.cw{perspective:700px;width:84px;height:84px;flex:none}
.cube{position:relative;width:84px;height:84px;transform-style:preserve-3d;animation:cube 9s linear infinite;will-change:transform}
.cube i{position:absolute;inset:0;border:1.5px solid var(--cube-edge);border-radius:8px;background:linear-gradient(135deg,var(--cube-a),var(--cube-b));box-shadow:inset 0 0 16px var(--cube-b)}
.cube i:nth-child(1){transform:translateZ(42px)}.cube i:nth-child(2){transform:rotateY(180deg) translateZ(42px)}
.cube i:nth-child(3){transform:rotateY(90deg) translateZ(42px)}.cube i:nth-child(4){transform:rotateY(-90deg) translateZ(42px)}
.cube i:nth-child(5){transform:rotateX(90deg) translateZ(42px)}.cube i:nth-child(6){transform:rotateX(-90deg) translateZ(42px)}
@keyframes cube{from{transform:rotateX(-22deg) rotateY(0)}to{transform:rotateX(-22deg) rotateY(360deg)}}
/* thinking */
.think-head{display:flex;align-items:center;gap:12px;padding:16px 20px;border-radius:20px;margin-bottom:10px}
.verbs{position:relative;flex:1;height:24px;font-size:15px;line-height:24px}
.verbs span{position:absolute;left:0;top:0;white-space:nowrap;opacity:0;background:linear-gradient(90deg,var(--muted) 0%,var(--ink) 40%,var(--muted) 80%);background-size:250% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent;animation:verb 10.4s linear infinite both,sweep 2.2s linear infinite;animation-delay:calc(var(--i)*2.6s),0s}
@keyframes verb{0%{opacity:0;transform:translateY(8px)}3%,22%{opacity:1;transform:none}25%,100%{opacity:0;transform:translateY(-8px)}}
@keyframes sweep{from{background-position:100% 0}to{background-position:-150% 0}}
.prog{height:3px;border-radius:3px;background:var(--hair);overflow:hidden;margin:0 6px 14px}
.prog i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),#eba184);transform-origin:0 50%;transform:scaleX(var(--p,.04));transition:transform .9s var(--ease)}
.trace{display:flex;flex-direction:column;gap:9px;padding:16px 20px;border-radius:20px;margin-bottom:14px}
.st{display:flex;align-items:flex-start;gap:10px;font-size:13.5px;line-height:1.5;color:var(--muted);animation:fadeUp .4s var(--ease) backwards;transition:color .3s}
.st i{flex:none;margin-top:7px;width:7px;height:7px;border-radius:50%;background:var(--soft);transition:background-color .3s,box-shadow .3s}
.st.act{color:var(--ink)}.st.act i{background:var(--accent);box-shadow:0 0 0 4px var(--accent-soft);animation:pulse 1.2s ease-in-out infinite}
.st em{font-style:normal;margin-left:8px;color:var(--soft);font-variant-numeric:tabular-nums}
@keyframes pulse{50%{opacity:.3}}
/* loading well: the sheen is a translated layer (compositor only), not an animated background */
.well{position:relative;overflow:hidden;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;height:260px;border-radius:20px;background:var(--well-a);border:1px solid var(--line);box-shadow:var(--well)}
.well::after{content:"";position:absolute;inset:0;z-index:0;background:linear-gradient(100deg,transparent 30%,var(--well-b) 50%,transparent 70%);transform:translateX(-100%);animation:sheen 1.9s linear infinite;pointer-events:none}
@keyframes sheen{to{transform:translateX(100%)}}
.wellmsg{font-size:13.5px;color:var(--muted)}
/* summary */
.sumhead{display:flex;align-items:center;justify-content:space-between;padding:2px 4px 12px}
.sumhead span:first-child{font-size:13px;font-weight:500;color:var(--muted)}
.chipv{display:inline-flex;align-items:center;gap:7px;padding:5px 12px;border-radius:999px;font-size:12.5px;font-weight:500;background:var(--card);border:1px solid var(--line);box-shadow:var(--e1);color:var(--good)}
.chipv::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}.chipv.low{color:var(--warn)}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:16px}
.stat{position:relative;padding:20px 22px 18px;border-radius:20px;overflow:hidden;animation:fadeUp .6s var(--ease) backwards;animation-delay:calc(300ms + var(--d,0ms))}
.big{display:flex;align-items:flex-start;font-family:var(--serif);font-size:clamp(34px,4vw,56px);font-weight:500;letter-spacing:-.03em;line-height:1;font-variant-numeric:lining-nums tabular-nums}
.big sup{font-family:var(--sans);font-size:.3em;font-weight:500;letter-spacing:0;margin:.35em 0 0 4px;color:var(--soft)}
@property --n{syntax:"<integer>";initial-value:0;inherits:false}
.num{counter-reset:n var(--n);animation:count 1.5s var(--ease) both;animation-delay:calc(300ms + var(--d,0ms));display:inline-block}
.num::after{content:counter(n)}
@keyframes count{from{--n:0}to{--n:var(--to)}}
.slab{margin-top:14px;font-size:14.5px;font-weight:500}.ssub{margin-top:3px;font-size:12.5px;color:var(--muted)}
.stat.hero{grid-column:1/-1;padding:26px 30px 24px;border:0;background:linear-gradient(135deg,#ea9374 0%,#d97757 45%,#bb5738 100%);box-shadow:0 2px 4px rgba(187,87,56,.22),0 20px 40px -14px rgba(187,87,56,.5),inset 0 1px 0 rgba(255,255,255,.35)}
.stat.hero::after{content:"";position:absolute;right:-70px;top:-90px;width:300px;height:300px;border-radius:50%;background:radial-gradient(circle,rgba(255,255,255,.28),transparent 65%);pointer-events:none}
.stat.hero .big{font-size:clamp(66px,9.5vw,112px);color:#fff}.stat.hero .big sup{color:rgba(255,255,255,.75)}
.stat.hero .slab{font-size:18px;color:#fff;margin-top:8px}.stat.hero .ssub{font-size:13.5px;color:rgba(255,255,255,.88)}
@media(max-width:760px){.stats{grid-template-columns:1fr 1fr}}
/* viewer + media */
model-viewer{width:100%;height:460px;border-radius:16px;background:radial-gradient(circle at 50% 30%,var(--view-a) 0%,var(--view-b) 70%,var(--view-c) 100%);box-shadow:inset 0 0 0 1px var(--line);--poster-color:transparent;--progress-bar-color:var(--accent);display:block}
.dl a{color:var(--ink);text-decoration:none;border:1px solid var(--line);background:var(--card);border-radius:999px;padding:7px 14px;font-size:13px;font-weight:450;box-shadow:var(--e1);transition:transform .25s var(--ease),box-shadow .25s var(--ease),color .2s}
.dl a:hover{color:var(--accent-ink);transform:translateY(-1px);box-shadow:var(--e2)}
.note{background:var(--accent-soft);border-radius:12px;padding:10px 13px;font-size:13px;margin-top:10px;line-height:1.5}
.err{background:var(--warn-soft);color:var(--warn);border:1px solid var(--line);border-radius:16px;padding:14px 16px;font-size:13.5px;margin-bottom:16px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.kpi{background:var(--tile);border:1px solid var(--line);border-radius:14px;padding:12px 14px}
.kpi b{display:block;font-size:20px;font-weight:500;letter-spacing:-.01em;font-variant-numeric:tabular-nums}.kpi span{font-size:12px;color:var(--soft)}
.ok{color:var(--good)}.bad{color:var(--warn)}
.strip{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.cands{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
figure{margin:0;position:relative}
.strip img,.cands img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:14px;display:block;border:1px solid var(--line);background:radial-gradient(circle at 50% 30%,var(--view-a),var(--view-c))}
.cands img{opacity:0;transform:scale(.97);transition:opacity .55s var(--ease),transform .6s var(--ease),outline-color .3s;transition-delay:calc(var(--k,0)*90ms)}
.cands img.in{opacity:1;transform:none}
figcaption{font-size:12px;color:var(--muted);margin-top:7px}
.cands figure.win img{outline:2px solid var(--accent);outline-offset:3px}
.tag{position:absolute;top:9px;left:9px;background:var(--accent);color:#fff;font-size:11px;font-weight:500;padding:3px 9px;border-radius:999px;opacity:0;transform:translateY(-4px);transition:opacity .3s var(--ease),transform .3s var(--ease)}
.win .tag{opacity:1;transform:none}
.mt{font-size:11.5px;color:var(--soft);margin-top:6px;display:flex;justify-content:space-between;font-variant-numeric:tabular-nums}
.bar{height:4px;background:var(--hair);border-radius:3px;margin-top:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent);border-radius:3px}
.tm{display:grid;grid-template-columns:minmax(120px,1.2fr) 3fr 56px;gap:12px;align-items:center;padding:6px 0;font-size:13px}
.tm .tb{height:8px;border-radius:5px;background:var(--hair);overflow:hidden}.tm .tb i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),#bb5738);border-radius:5px;transform-origin:0 50%;animation:grow .9s var(--ease) backwards}
@keyframes grow{from{transform:scaleX(0)}}
.tm span:last-child{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
.hidden{display:none!important}
@media(max-width:640px){.kpis,.strip{grid-template-columns:1fr 1fr}.cands{grid-template-columns:1fr 1fr}.idle ol{grid-template-columns:1fr}}
/* floating cubes + cube loader */
.fl{position:absolute;inset:0;overflow:hidden;pointer-events:none;z-index:0}
.fc{position:absolute;left:var(--x,0);top:var(--y,0);width:var(--s);height:var(--s);perspective:420px;animation:bob var(--t,4s) ease-in-out infinite;animation-delay:var(--dl,0s);opacity:var(--o,.8);will-change:transform}
.fc b{position:absolute;inset:0;display:block;transform-style:preserve-3d;animation:tum var(--r,9s) linear infinite;animation-delay:var(--dl,0s);will-change:transform}
.fc b i{position:absolute;inset:0;border:1px solid var(--cube-edge);border-radius:3px;background:linear-gradient(135deg,var(--cube-a),var(--cube-b))}
.fc b i:nth-child(1){transform:translateZ(calc(var(--s)/2))}.fc b i:nth-child(2){transform:rotateY(180deg) translateZ(calc(var(--s)/2))}
.fc b i:nth-child(3){transform:rotateY(90deg) translateZ(calc(var(--s)/2))}.fc b i:nth-child(4){transform:rotateY(-90deg) translateZ(calc(var(--s)/2))}
.fc b i:nth-child(5){transform:rotateX(90deg) translateZ(calc(var(--s)/2))}.fc b i:nth-child(6){transform:rotateX(-90deg) translateZ(calc(var(--s)/2))}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-18px)}}
@keyframes tum{from{transform:rotateX(20deg) rotateY(0)}to{transform:rotateX(380deg) rotateY(360deg)}}
.idle>*:not(.fl),.well>*:not(.fl){position:relative;z-index:1}
.idle .fc{opacity:.38!important}
.loader{display:flex;align-items:center;gap:11px;width:56px;height:24px;flex:none}
.loader .fc{position:relative;left:0;top:0;opacity:1;perspective:200px;animation-name:bobs}
@keyframes bobs{0%,100%{transform:translateY(3px)}50%{transform:translateY(-3px)}}
/* stage tiles */
.stepper{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:14px}
.sg{padding:10px 12px;border-radius:12px;background:var(--card);border:1px solid var(--line);box-shadow:var(--e1);font-size:12.5px;color:var(--soft);transition:color .4s var(--ease),border-color .4s var(--ease),box-shadow .4s var(--ease)}
.sg b{display:block;font-weight:500;font-size:11px;margin-bottom:3px;font-variant-numeric:tabular-nums}
.sg.done{color:var(--good)}.sg.act{color:var(--ink);border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft),var(--e1)}
@media(max-width:640px){.stepper{gap:4px}.sg{padding:8px 6px;font-size:11px}}
/* identified as */
.idrow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.idname{font-family:var(--serif);font-size:clamp(26px,3.2vw,36px);font-weight:500;letter-spacing:-.02em;line-height:1.15;text-transform:capitalize}
.srcb{font-size:12px;font-weight:500;padding:4px 11px;border-radius:999px;background:var(--accent-soft);color:var(--accent-ink)}
.idcap{margin:10px 0 0;font-size:14.5px;color:var(--muted);line-height:1.6;max-width:580px}
.idmeta{margin-top:10px;font-size:12.5px;color:var(--soft)}.idmeta b{font-weight:500;color:var(--muted)}
.alt small{opacity:.7;margin-left:3px}
.idhelp{margin-top:10px;font-size:12px;color:var(--soft)}
.bars{display:grid;gap:7px;margin-top:16px;max-width:540px}
.bar{position:relative;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 13px;border-radius:12px;border:1px solid var(--line);background:var(--tile);font:inherit;font-size:13.5px;color:var(--ink);text-align:left;text-transform:capitalize;cursor:pointer;overflow:hidden;transition:transform .2s var(--ease),border-color .2s}
.bar i{position:absolute;inset:0 auto 0 0;width:calc(var(--w)*1%);min-width:3px;background:var(--accent-soft);transition:width .8s var(--ease)}
.bar span,.bar small{position:relative}
.bar small{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;text-transform:none}
.bar:hover{transform:translateY(-1px);border-color:var(--accent)}
.bar.on{border-color:var(--accent)}.bar.on span{font-weight:500;color:var(--accent-ink)}
.srcb.low{background:var(--warn-soft);color:var(--warn)}
.warnline{margin-top:12px;padding:9px 12px;border-radius:12px;background:var(--warn-soft);color:var(--warn);font-size:12.5px;line-height:1.45;overflow-wrap:anywhere}
/* ---------- input: Draw / Upload pill ---------- */
.modebar{display:flex;align-items:center;justify-content:space-between;gap:10px 14px;flex-wrap:wrap;margin-bottom:14px}
.modehint{font-size:12px;color:var(--soft)}
.mode{position:relative;display:inline-grid;grid-template-columns:repeat(3,1fr);padding:4px;border-radius:999px;background:var(--recess);border:1px solid var(--line);box-shadow:var(--well)}
.mode::before{content:"";position:absolute;top:4px;bottom:4px;left:4px;width:calc((100% - 8px)/3);border-radius:999px;background:var(--btn);box-shadow:var(--btn-sh);
transform:translateX(calc(var(--pos,0)*100%));transition:transform .55s var(--spring)}
.mode button{position:relative;z-index:1;display:inline-flex;align-items:center;justify-content:center;gap:8px;min-width:84px;padding:9px 14px;border:0;border-radius:999px;background:transparent;
color:var(--muted);font-size:13.5px;font-weight:500;cursor:pointer;transition:color .3s var(--ease)}
.mode button:not(.on):hover{color:var(--ink)}.mode button.on{color:var(--btn-text)}
.mode svg,.pill svg{flex:none}
.pane.enter{animation:paneIn .4s var(--ease) backwards}
@keyframes paneIn{from{opacity:0;transform:translateY(8px) scale(.992)}to{opacity:1;transform:none}}
/* drawing pad */
.padwrap{position:relative;border-radius:20px;overflow:hidden;line-height:0;background:#fff}
.padwrap::after{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;box-shadow:inset 0 1px 3px rgba(21,21,26,.12),inset 0 0 0 1px var(--hair)}
#pad{border-radius:0;box-shadow:none}
.padwrap.fine #pad{cursor:none}
.padhint{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;text-align:center;pointer-events:none;line-height:1.45;font-size:12.5px;
color:rgba(21,21,26,.42);opacity:0;transform:translateY(8px);transition:opacity .5s var(--ease),transform .6s var(--ease)}
.padhint b{font-size:15px;font-weight:500;color:rgba(21,21,26,.55)}
.padwrap.empty .padhint{opacity:1;transform:none}
.ring{position:absolute;left:0;top:0;border-radius:50%;pointer-events:none;z-index:2;border:1.5px solid rgba(21,21,26,.75);box-shadow:0 0 0 1px rgba(255,255,255,.9);opacity:0;transition:opacity .2s;will-change:transform}
.ring.er{border-style:dashed}
.tools{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:14px}
.tg{display:inline-flex;gap:2px;padding:4px;border-radius:999px;background:var(--recess);border:1px solid var(--line);box-shadow:var(--well)}
.tg .pill{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-color:transparent;background:transparent;box-shadow:none}
.tg .pill:hover{transform:none;box-shadow:none}
.tg .pill.on{background:var(--btn);color:var(--btn-text);box-shadow:var(--btn-sh)}
.pill:disabled{opacity:.38;cursor:default;pointer-events:none}
.sz{display:flex;align-items:center;gap:8px;margin-left:auto;font-size:12.5px;color:var(--muted)}
.sz input{width:104px}.sz b{min-width:20px;text-align:right;font-weight:500;color:var(--ink);font-variant-numeric:tabular-nums}
.scrollpill{align-items:center;gap:6px}
/* upload */
.drop{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;aspect-ratio:1;padding:24px;border-radius:20px;text-align:center;cursor:pointer;
border:1.5px dashed var(--hair);background:var(--tile);transition:border-color .3s,background-color .3s}
.drop:hover,.drop:focus-visible,.card.dragging .drop{border-color:var(--accent);background:var(--accent-soft)}
.dicon{width:62px;height:62px;border-radius:50%;display:grid;place-items:center;background:var(--card);color:var(--accent-ink);box-shadow:var(--e2);transition:transform .55s var(--spring)}
.drop:hover .dicon,.card.dragging .dicon{transform:translateY(-5px) scale(1.07)}
.drop b{font-size:15px;font-weight:500}.drop span{max-width:270px;font-size:12.5px;line-height:1.55;color:var(--soft)}
.upprev img{display:block;width:100%;aspect-ratio:1;object-fit:contain;border-radius:20px;background:#fff;box-shadow:var(--well),0 0 0 1px var(--hair)}
.upprev .row{justify-content:center}
.card.dragging{box-shadow:0 0 0 2px var(--accent),var(--e2)}
/* results: summary first, the 3D model right under it, the rest below (applied once the run is done) */
.stage.fin{display:flex;flex-direction:column}
.stage.fin #res{order:1}.stage.fin #top{order:2}
.stage.fin #top.enter>*{animation-delay:calc(560ms + var(--k,0)*90ms)}
#res{scroll-margin-top:16px}
#mv{opacity:0;transform:scale(.985);transition:opacity .9s var(--ease),transform 1s var(--ease)}
#mv.ready{opacity:1;transform:none}
#kpis .kpi,.dl a,.strip figure{animation:fadeUp .6s var(--ease) backwards;animation-delay:calc(560ms + var(--k,0)*55ms)}
.strip figure{animation-delay:calc(var(--k,0)*90ms)}
#res .strip figure{animation-delay:calc(700ms + var(--k,0)*90ms)}
#res .note{animation:fadeIn .8s var(--ease) .7s backwards}
/* theme change cross-fades surfaces (class is on only for a moment) */
html.tx *,html.tx *::before,html.tx *::after{transition-property:background-color,color,border-color,box-shadow,transform,opacity!important;transition-duration:.35s!important;transition-timing-function:var(--ease)!important}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}

/* ---- 4 photos ---- */
.phgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.phslot{position:relative;animation:fadeUp .5s var(--ease) backwards}
.phslot:nth-child(2){animation-delay:70ms}.phslot:nth-child(3){animation-delay:140ms}.phslot:nth-child(4){animation-delay:210ms}
.phdrop{position:relative;display:flex;align-items:center;justify-content:center;aspect-ratio:1;border-radius:18px;overflow:hidden;cursor:pointer;border:1.5px dashed var(--hair);background:var(--tile);
transition:border-color .3s,background-color .3s,transform .4s var(--spring),box-shadow .3s var(--ease)}
.phdrop:hover,.phdrop:focus-visible,.phslot.drag .phdrop{border-color:var(--accent);background:var(--accent-soft);transform:translateY(-2px)}
.phslot.has .phdrop{border-style:solid;border-color:var(--line);background:#fff;box-shadow:var(--e1)}
.phdrop img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#fff;animation:fadeIn .5s var(--ease)}
.phname{position:absolute;left:10px;top:9px;z-index:2;font-size:11.5px;font-weight:500;padding:3px 10px;border-radius:999px;background:var(--card);color:var(--muted);box-shadow:var(--e1)}
.phplus{font-size:34px;font-weight:300;color:var(--soft);line-height:1;transition:transform .5s var(--spring)}
.phdrop:hover .phplus{transform:rotate(90deg) scale(1.15);color:var(--accent-ink)}
.phslot.has .phplus{display:none}
.phx{position:absolute;right:8px;top:8px;z-index:3;width:26px;height:26px;border-radius:50%;border:0;background:var(--btn);color:var(--btn-text);font-size:17px;line-height:1;cursor:pointer;box-shadow:var(--btn-sh);transition:transform .3s var(--spring)}
.phx:hover{transform:scale(1.15)}
/* ---- result hero ---- */
.hero2{position:relative;grid-column:1/-1;overflow:hidden;display:flex;align-items:center;justify-content:space-between;gap:22px;flex-wrap:wrap;padding:30px 34px;border-radius:24px;color:#fff;
background:linear-gradient(135deg,#ee9a7c 0%,#d97757 42%,#b3522f 100%);box-shadow:0 2px 4px rgba(187,87,56,.22),0 22px 44px -16px rgba(187,87,56,.55),inset 0 1px 0 rgba(255,255,255,.35);animation:fadeUp .7s var(--ease) .15s backwards}
.hero2::after{content:"";position:absolute;right:-80px;top:-100px;width:340px;height:340px;border-radius:50%;background:radial-gradient(circle,rgba(255,255,255,.3),transparent 65%);pointer-events:none}
.hero2 .fl{opacity:.55}.hero2 .fc b i{border-color:rgba(255,255,255,.7);background:linear-gradient(135deg,rgba(255,255,255,.25),rgba(255,255,255,.05))}
.hero2>*:not(.fl){position:relative;z-index:1}
.h2txt{flex:1 1 260px;min-width:0}
.h2eye{display:inline-block;font-size:12.5px;font-weight:500;letter-spacing:.02em;opacity:.9;animation:fadeUp .6s var(--ease) .35s backwards}
.h2name{margin:6px 0 14px;font-family:var(--serif);font-size:clamp(40px,6.4vw,84px);font-weight:500;letter-spacing:-.035em;line-height:1;text-transform:capitalize;overflow-wrap:anywhere;animation:fadeUp .7s var(--ease) .42s backwards}
.h2chips{display:flex;flex-wrap:wrap;gap:8px}
.h2chip{padding:6px 14px;border-radius:999px;font-size:12.5px;font-weight:500;background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.35);backdrop-filter:blur(6px);animation:pop .6s var(--spring) backwards}
.h2chip:nth-child(1){animation-delay:.6s}.h2chip:nth-child(2){animation-delay:.7s}.h2chip:nth-child(3){animation-delay:.8s}.h2chip:nth-child(4){animation-delay:.9s}
.h2art{display:flex;align-items:center;gap:14px;flex:none}
.h2art figure{width:clamp(96px,13vw,150px);margin:0}
.h2art img{display:block;width:100%;aspect-ratio:1;object-fit:contain;border-radius:18px;background:#fff;border:3px solid #fff;box-shadow:0 10px 26px -8px rgba(60,20,5,.55)}
.h2in{transform:rotate(-4deg);animation:fadeUp .7s var(--ease) .5s backwards}
.h2out{transform:rotate(3deg);animation:popOut .9s var(--spring) 1s backwards}
.h2art figcaption{margin-top:9px;text-align:center;font-size:12px;color:rgba(255,255,255,.92)}
.h2arrow{flex:none;color:#fff;opacity:.95;stroke-dasharray:90;animation:draw .8s var(--ease) .85s backwards}
@keyframes draw{from{stroke-dashoffset:90;opacity:0}to{stroke-dashoffset:0}}
@keyframes popOut{from{opacity:0;transform:scale(.55) rotate(12deg)}to{opacity:1;transform:rotate(3deg)}}
.tile{padding:18px 20px 16px}
.tile .ti{display:grid;place-items:center;width:36px;height:36px;margin-bottom:14px;border-radius:12px;background:var(--accent-soft);color:var(--accent-ink);transition:transform .5s var(--spring)}
.tile:hover .ti{transform:rotate(-8deg) scale(1.12)}
.tile .big{font-size:clamp(28px,3.2vw,44px)}
.meter{display:inline-flex;align-items:flex-end;gap:5px;height:34px}
.meter i{width:12px;border-radius:4px;background:var(--hair);height:calc(10px + var(--i)*8px);transform-origin:50% 100%;animation:grow2 .6s var(--spring) calc(.8s + var(--i)*.12s) backwards}
.meter i.on{background:linear-gradient(180deg,#eba184,var(--accent))}
@keyframes grow2{from{transform:scaleY(0)}}
@media(max-width:640px){.hero2{padding:22px 20px}.h2art{width:100%;justify-content:center}}
/* ---- palette ---- */
.palette{display:flex;flex-wrap:wrap;align-items:center;gap:9px;margin-top:14px}
.phint{font-size:12.5px;color:var(--muted);margin-right:4px}
.sw{width:26px;height:26px;padding:0;border-radius:50%;border:2px solid var(--card);background:var(--c);cursor:pointer;box-shadow:0 0 0 1px var(--line),var(--e1);
transition:transform .4s var(--spring),box-shadow .3s var(--ease)}
.sw:hover{transform:translateY(-2px) scale(1.14)}
.sw.on{transform:scale(1.22);box-shadow:0 0 0 2px var(--c),0 0 0 6px var(--accent-soft)}
/* ---- bigger name text in the summary bar ---- */
.bigt{display:block;font-family:var(--serif);font-size:clamp(22px,2.4vw,32px);font-weight:500;letter-spacing:-.02em;line-height:1.1;text-transform:capitalize;overflow-wrap:anywhere}
/* ---- more motion ---- */
.card:not(#inputcard){transition:transform .4s var(--ease),box-shadow .4s var(--ease)}
.card:not(#inputcard):hover{transform:translateY(-2px);box-shadow:var(--e3)}
.stat{transition:transform .4s var(--spring),box-shadow .4s var(--ease)}
.stat:hover{transform:translateY(-4px);box-shadow:var(--e3)}
.chipv{animation:pop .7s var(--spring) .5s backwards}
@keyframes pop{from{opacity:0;transform:scale(.6)}to{opacity:1;transform:none}}
button.go{background-image:linear-gradient(110deg,transparent 35%,rgba(255,255,255,.2) 50%,transparent 65%);background-size:250% 100%;background-position:150% 0;
transition:transform .3s var(--ease),background-color .25s,opacity .25s,box-shadow .3s var(--ease),background-position .9s var(--ease)}
button.go:hover{background-position:-50% 0}
button.go:disabled{animation:breathe 1.4s ease-in-out infinite}
@keyframes breathe{50%{opacity:.8}}
.prog i{position:relative;overflow:hidden}
.prog i::after{content:"";position:absolute;inset:0;background:linear-gradient(100deg,transparent,rgba(255,255,255,.55),transparent);transform:translateX(-100%);animation:sheen 1.4s linear infinite}
.strip img,.cands img{transition:opacity .55s var(--ease),transform .5s var(--ease),outline-color .3s}
.strip figure:hover img,.cands figure:hover img.in{transform:scale(1.035)}
.kpi{transition:transform .35s var(--spring),border-color .3s}.kpi:hover{transform:translateY(-3px);border-color:var(--accent)}
.note{transition:transform .35s var(--ease)}.note:hover{transform:translateX(3px)}
.idname{animation:fadeUp .6s var(--ease) backwards}
.seg button,.pill,.dl a,.chip{transition:transform .3s var(--spring),box-shadow .25s var(--ease),color .2s,background-color .25s,border-color .25s}
.seg button:active{transform:scale(.97)}
</style></head><body><div class="wrap">
<header>
 <div class="brand"><div class="mark"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"><path d="M12 2.5 20.5 7v10L12 21.5 3.5 17V7z"/><path d="M3.5 7 12 12l8.5-5M12 12v9.5"/></svg></div>
  <div><h1>dribble<b>3d</b></h1><p>Sketch it. Get a 3D model you can print.</p></div></div>
 <div class="switch" role="group" aria-label="Theme">
  <button type="button" data-t="light" title="Light" aria-label="Light"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></button>
  <button type="button" data-t="dark" title="Dark" aria-label="Dark"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg></button>
  <button type="button" data-t="auto" title="Match system" aria-label="Match system"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/></svg></button>
 </div>
</header>
<div class="grid">
<div class="left">
 <div class="card reveal" id="inputcard" style="--k:0">
  <div class="modebar">
   <div class="mode" id="inmode" role="tablist" aria-label="Input method">
    <button type="button" class="on" data-v="draw" role="tab"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>Draw</button>
    <button type="button" data-v="upload" role="tab"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>Upload</button>
    <button type="button" data-v="photos" role="tab"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>4 Photos</button>
   </div>
   <span class="modehint" id="modehint">Sketch it or bring a photo</span>
  </div>
  <div class="pane" id="drawpane">
   <div class="padwrap empty" id="padwrap">
    <canvas id="pad" width="768" height="768"></canvas>
    <div class="padhint"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg><b>Draw one object</b><span>Big and simple. A rough outline is fine.</span></div>
    <i class="ring" id="ring"></i>
   </div>
   <div class="palette" id="palette" role="group" aria-label="Ink colour"><span class="phint">Ink</span></div>
   <div class="tools">
    <div class="tg">
     <button class="pill on" id="pen" type="button" title="Pen (B)"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>Pen</button>
     <button class="pill" id="eraser" type="button" title="Eraser (E)"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m7 21-4.3-4.3a1 1 0 0 1 0-1.4l9.6-9.6a1 1 0 0 1 1.4 0l5.6 5.6a1 1 0 0 1 0 1.4L13 21"/><path d="M22 21H7"/><path d="m5 11 9 9"/></svg>Eraser</button>
    </div>
    <div class="tg">
     <button class="pill" id="undo" type="button" title="Undo (Ctrl+Z)" aria-label="Undo" disabled><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>Undo</button>
     <button class="pill" id="redo" type="button" title="Redo (Ctrl+Shift+Z)" aria-label="Redo" disabled><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 14 5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13"/></svg>Redo</button>
     <button class="pill" id="clear" type="button" title="Clear the canvas"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>Clear</button>
    </div>
    <button class="pill scrollpill" id="scrollmode" type="button" title="Let a finger swipe scroll the page instead of drawing">Scroll mode</button>
    <label class="sz" title="Brush size ( [ and ] )">Size<input type="range" id="size" min="2" max="24" value="7" aria-label="Brush size"><b id="sizev">7</b></label>
   </div>
  </div>
  <div class="pane hidden" id="photopane">
   <div class="phgrid" id="phgrid"></div>
   <div class="hint" id="phmsg">Walk around ONE object: front, left, back, right. Same distance, plain background, whole object in frame. Front plus one more angle is the minimum.</div>
  </div>
  <div class="pane hidden" id="uploadpane">
   <label class="drop" id="drop" tabindex="0">
    <input type="file" id="file" accept="image/*" hidden>
    <span class="dicon"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg></span>
    <b>Drop an image here</b>
    <span>or click to browse. You can also paste one with Ctrl+V. A clean photo or a drawing both work.</span>
   </label>
   <div class="upprev hidden" id="upprev">
    <img id="upimg" alt="Uploaded image">
    <div class="row"><button class="pill" id="upreplace" type="button">Replace</button><button class="pill" id="uptodraw" type="button">Draw on it</button><button class="pill" id="upremove" type="button">Remove</button></div>
   </div>
   <div class="hint" id="upmsg"></div>
  </div>
 </div>
 <div class="card reveal" style="--k:1">
  <div class="two"><div><span class="lab">What is it? (optional)</span><input type="text" id="name" placeholder="e.g. blue car"></div>
  <div><span class="lab">Input type</span><select id="mode"><option value="auto">Auto-detect</option><option value="sketch">Sketch</option><option value="photo">Photo</option></select></div></div>
  <span class="lab" style="margin-top:16px">Refine (keywords, colours, materials)</span>
  <input type="text" id="fb" placeholder="e.g. thinner, blue, carved wooden, metallic">
  <div class="row" id="chips" style="margin-top:10px"></div>
  <span class="lab" style="margin-top:18px">Speed</span>
  <div class="seg" id="speed">
   <button type="button" data-v="fast">Fast<small>3 photoreal candidates</small></button>
   <button type="button" class="on" data-v="quality">Quality<small>5 candidates, finest mesh, slower</small></button>
  </div>
  <span class="lab" style="margin-top:18px">Geometry</span>
  <div class="seg" id="geom">
   <button type="button" data-v="single">Single view<small>TripoSR, fastest, flat depth</small></button>
   <button type="button" class="on" data-v="multi">Multi-view<small>Zero123++ views → Hunyuan3D-2mv, best shape</small></button>
  </div>
  <div class="hint hidden" id="geomnote"></div>
  <label class="chk"><input type="checkbox" id="cpu"> Instant CPU mode (no GPU, under a second)</label>
  <button class="go" id="go" type="button">Generate 3D</button>
 </div>
</div>
<div class="stage">
 <div class="idle reveal" id="idle" style="--k:2">
  <div class="fl" data-n="9" data-x0="80" data-xs="15"></div>
  <div class="art"><div class="cw"><div class="cube"><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
   <div><h3>Your model shows up here</h3><p>Draw anything on the left, or drop in a photo. You will first see what it was recognised as and the generated images, then the 3D model. Say a colour (try “blue car”) and it carries through.</p></div></div>
  <ol><li><b>01</b> Read your lines</li><li><b>02</b> Name the object</li><li><b>03</b> Render it clean</li><li><b>04</b> Build the 3D model</li></ol>
 </div>
 <div id="busy" class="hidden">
  <div class="think-head"><div class="loader" id="loader"></div>
   <div class="verbs"><span style="--i:0">Reading your lines</span><span style="--i:1">Naming the object</span><span style="--i:2">Rendering it clean</span><span style="--i:3">Building the 3D model</span></div></div>
  <div class="prog"><i id="progbar"></i></div>
  <div class="stepper" id="stepper"></div>
 </div>
 <div id="err" class="err hidden"></div>
 <div id="top" class="hidden">
  <div class="card ident" id="identcard"></div>
  <div class="card hidden" id="mvcard"><span class="lab">Views sent to the 3D builder</span><div class="strip" id="mvs"></div></div>
  <div class="card hidden" id="candcard"><span class="lab" id="candlab">Generated images</span><div class="cands" id="cands"></div></div>
 </div>
 <div id="busy2" class="hidden">
  <div class="trace" id="steps"></div>
  <div class="well"><div class="fl" data-n="9"></div><div class="cw"><div class="cube"><i></i><i></i><i></i><i></i><i></i><i></i></div></div><div class="wellmsg" id="wellmsg">Working on it…</div></div>
 </div>
 <div id="res" class="hidden">
  <div class="sumhead"><span>Summary</span><span class="chipv" id="chip"></span></div>
  <div class="stats" id="stats"></div>
  <div class="card"><span class="lab">3D model</span>
   <model-viewer id="mv" camera-controls auto-rotate shadow-intensity="1" exposure="1.05" environment-image="neutral" interaction-prompt="none"></model-viewer>
   <div class="row dl" id="dl"></div>
   <div class="row"><span class="lab" style="margin:0 6px 0 0">Material</span><span id="presets" style="display:contents"></span></div>
   <div id="notes"></div></div>
  <div class="card"><span class="lab">Under the hood: closed-loop check and mesh health</span><div class="kpis" id="kpis"></div><div class="hint">Shape overlap is only a rough self-check of how closely the 3D outline traces your sketch. Loose or stylised sketches score lower even when the model looks right, so judge by the model above.</div></div>
  <div class="card"><span class="lab">Views rendered from the mesh</span><div class="strip" id="views"></div></div>
  <div class="card"><span class="lab">Pipeline</span><div id="tim"></div></div>
 </div>
</div></div></div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// ---- show / hide with a short fade (re-shows restart the entrance animation)
function put(el,on){
 if(!el)return;clearTimeout(el._h);
 if(on){if(el.classList.contains('hidden')||el.classList.contains('leave')){el.classList.remove('hidden','leave','enter');void el.offsetWidth;el.classList.add('enter')}}
 else if(!el.classList.contains('hidden')){el.classList.add('leave');el._h=setTimeout(()=>{el.classList.add('hidden');el.classList.remove('leave','enter')},240)}}
// ---- theme: light / dark / auto (surfaces cross-fade for a moment)
const root=document.documentElement;
function setTheme(p){const m=p==='auto'?(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'):p;
 root.classList.add('tx');root.dataset.theme=m;root.dataset.pref=p;try{localStorage.setItem('d3t',p)}catch(e){}
 clearTimeout(setTheme.t);setTheme.t=setTimeout(()=>root.classList.remove('tx'),450)}
document.querySelectorAll('.switch button').forEach(b=>b.onclick=()=>setTheme(b.dataset.t));
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if(root.dataset.pref==='auto')setTheme('auto')});
// ---- input: Draw / Upload pill + drawing pad
const cv=$('#pad'),ctx=cv.getContext('2d',{willReadFrequently:true}),wrapEl=$('#padwrap'),ring=$('#ring'),sizeEl=$('#size');
let inkCol='#111111',drawing=false,erase=false,hist=[],redoS=[],blankNow=true,lastP=null,prevMid=null,curW=0,mode='draw',uploaded=false;
const fine=matchMedia('(pointer:fine)').matches;if(fine)wrapEl.classList.add('fine');
function blank(){ctx.fillStyle='#fff';ctx.fillRect(0,0,cv.width,cv.height)}blank();
function isBlank(){const d=ctx.getImageData(0,0,cv.width,cv.height).data;for(let i=0;i<d.length;i+=16){if(Math.min(d[i],d[i+1],d[i+2])<245)return false}return true}
function setBlank(b){blankNow=b;wrapEl.classList.toggle('empty',b)}
function pos(e){const r=cv.getBoundingClientRect();return[(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height]}
function syncBtns(){$('#undo').disabled=!hist.length;$('#redo').disabled=!redoS.length}
function snap(){hist.push(ctx.getImageData(0,0,cv.width,cv.height));if(hist.length>40)hist.shift();redoS=[];syncBtns()}
function brush(){return(+sizeEl.value)*(erase?3:1.6)}
function widthOf(e){return erase?brush():brush()*(e.pointerType==='pen'?.35+1.1*(e.pressure||.5):1)}
function ink(){ctx.fillStyle=ctx.strokeStyle=erase?'#fff':inkCol;ctx.lineCap=ctx.lineJoin='round'}
function dot(p,w){ink();ctx.beginPath();ctx.arc(p[0],p[1],w/2,0,6.2832);ctx.fill()}
function curve(p,w){const mid=[(lastP[0]+p[0])/2,(lastP[1]+p[1])/2];ink();ctx.lineWidth=w;ctx.beginPath();ctx.moveTo(prevMid[0],prevMid[1]);ctx.quadraticCurveTo(lastP[0],lastP[1],mid[0],mid[1]);ctx.stroke();prevMid=mid;lastP=p}
function moveRing(e){
 if(!fine||e.pointerType==='touch'){ring.style.opacity=0;return}
 const r=cv.getBoundingClientRect(),w=Math.max(6,brush()*r.width/cv.width);
 ring.style.width=ring.style.height=w+'px';ring.style.transform=`translate(${e.clientX-r.left-w/2}px,${e.clientY-r.top-w/2}px)`;ring.style.opacity=1;ring.classList.toggle('er',erase)}
cv.onpointerdown=e=>{
 if(e.pointerType==='touch'&&cv.classList.contains('scroll'))return;if(e.button>0)return;
 cv.setPointerCapture(e.pointerId);snap();drawing=true;setBlank(false);lastP=prevMid=pos(e);curW=widthOf(e);dot(lastP,curW);moveRing(e);e.preventDefault()};
cv.onpointermove=e=>{
 moveRing(e);if(!drawing)return;
 const evs=e.getCoalescedEvents?e.getCoalescedEvents():[];
 (evs.length?evs:[e]).forEach(ev=>{const p=pos(ev);if(Math.hypot(p[0]-lastP[0],p[1]-lastP[1])<1.5)return;curW=curW*.65+widthOf(ev)*.35;curve(p,curW)})};
function endDraw(){
 if(!drawing)return;drawing=false;ink();ctx.lineWidth=curW;ctx.beginPath();ctx.moveTo(prevMid[0],prevMid[1]);ctx.lineTo(lastP[0],lastP[1]);ctx.stroke();
 if(erase)setBlank(isBlank())}
cv.onpointerup=cv.onpointercancel=endDraw;
cv.onpointerleave=()=>{ring.style.opacity=0};
$('#scrollmode').onclick=()=>{const on=cv.classList.toggle('scroll');$('#scrollmode').classList.toggle('on',on)};
function setTool(er){erase=er;$('#pen').classList.toggle('on',!er);$('#eraser').classList.toggle('on',er)}
$('#pen').onclick=()=>setTool(false);$('#eraser').onclick=()=>setTool(true);
const INKS=[['black','#111111'],['red','#cd3030'],['orange','#f08c28'],['yellow','#f4ce3c'],['green','#46a050'],['blue','#346edc'],['purple','#8250be'],['pink','#f082aa'],['brown','#825537']];
INKS.forEach(([n,c],i)=>{const b=document.createElement('button');b.type='button';b.className='sw'+(i===0?' on':'');b.style.setProperty('--c',c);b.title=n;b.setAttribute('aria-label',n);
 b.onclick=()=>{inkCol=c;setTool(false);document.querySelectorAll('#palette .sw').forEach(x=>x.classList.toggle('on',x===b))};$('#palette').appendChild(b)});
function undo(){const h=hist.pop();if(!h)return;redoS.push(ctx.getImageData(0,0,cv.width,cv.height));ctx.putImageData(h,0,0);setBlank(isBlank());syncBtns()}
function redo(){const h=redoS.pop();if(!h)return;hist.push(ctx.getImageData(0,0,cv.width,cv.height));ctx.putImageData(h,0,0);setBlank(isBlank());syncBtns()}
function clearPad(){snap();blank();setBlank(true);uploaded=false}
$('#undo').onclick=undo;$('#redo').onclick=redo;$('#clear').onclick=clearPad;
sizeEl.oninput=()=>{$('#sizev').textContent=sizeEl.value;const r=cv.getBoundingClientRect(),w=Math.max(6,brush()*r.width/cv.width);
 ring.style.width=ring.style.height=w+'px';ring.style.transform=`translate(${r.width/2-w/2}px,${r.height/2-w/2}px)`;ring.style.opacity=1;ring.classList.toggle('er',erase);
 clearTimeout(sizeEl._t);sizeEl._t=setTimeout(()=>{ring.style.opacity=0},700)};
addEventListener('keydown',e=>{
 const t=e.target;if(t&&/^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)&&t.type!=='range')return;
 const k=e.key.toLowerCase(),m=e.ctrlKey||e.metaKey;
 if(m&&k==='z'){e.preventDefault();e.shiftKey?redo():undo()}else if(m&&k==='y'){e.preventDefault();redo()}
 else if(!m&&k==='b')setTool(false);else if(!m&&k==='e')setTool(true);
 else if(!m&&(k==='['||k===']')){sizeEl.value=Math.min(24,Math.max(2,+sizeEl.value+(k==='['?-2:2)));sizeEl.oninput()}});
// ---- Draw / Upload pill
function refreshUp(){const has=uploaded&&!blankNow;$('#drop').classList.toggle('hidden',has);$('#upprev').classList.toggle('hidden',!has);if(has)$('#upimg').src=cv.toDataURL('image/jpeg',.85)}
const MODES=['draw','upload','photos'],HINTS={draw:'Sketch it or bring a photo',upload:'Sketch it or bring a photo',photos:'Real photos from 4 sides'};
function setMode(m){
 const old=mode;mode=m;document.querySelectorAll('#inmode button').forEach(b=>b.classList.toggle('on',b.dataset.v===m));$('#inmode').style.setProperty('--pos',MODES.indexOf(m));
 $('#modehint').textContent=HINTS[m];$('#go').textContent=m==='photos'?'Build 3D from photos':'Generate 3D';
 MODES.forEach(k=>{const el=$('#'+(k==='draw'?'drawpane':k==='upload'?'uploadpane':'photopane'));if(k!==m){el.classList.add('hidden');el.classList.remove('enter')}});
 const on=$('#'+(m==='draw'?'drawpane':m==='upload'?'uploadpane':'photopane'));on.classList.remove('hidden','enter');void on.offsetWidth;on.classList.add('enter');
 if(m==='upload')refreshUp()}
document.querySelectorAll('#inmode button').forEach(b=>b.onclick=()=>{if(b.dataset.v!==mode)setMode(b.dataset.v)});
// ---- 4 photos
const PH=[['front','Front'],['left','Left side'],['back','Back'],['right','Right side']],photos=[null,null,null,null];
$('#phgrid').innerHTML=PH.map(([k,l],i)=>`<div class="phslot" data-i="${i}"><label class="phdrop" tabindex="0"><input type="file" accept="image/*" hidden><span class="phname">${l}</span><span class="phplus">+</span><img alt="${l} photo" class="hidden"></label><button type="button" class="phx hidden" aria-label="Remove ${l}">&times;</button></div>`).join('');
function phLoad(i,f){
 if(!f||!/^image\//.test(f.type))return;const url=URL.createObjectURL(f),im=new Image();
 im.onload=()=>{const s=Math.min(1,1024/Math.max(im.width,im.height)),c=document.createElement('canvas');c.width=Math.round(im.width*s);c.height=Math.round(im.height*s);
  const x=c.getContext('2d');x.fillStyle='#fff';x.fillRect(0,0,c.width,c.height);x.drawImage(im,0,0,c.width,c.height);URL.revokeObjectURL(url);
  photos[i]=c.toDataURL('image/jpeg',.9);phDraw(i)};im.src=url}
function phDraw(i){const sl=$('#phgrid').children[i],img=sl.querySelector('img'),has=!!photos[i];
 img.classList.toggle('hidden',!has);if(has)img.src=photos[i];sl.classList.toggle('has',has);sl.querySelector('.phx').classList.toggle('hidden',!has);
 const n=photos.filter(Boolean).length;$('#phmsg').textContent=n?`${n} of 4 photos added.`+(photos[0]?'':' A front photo is required.')+(n<4?' All 4 angles give the best result.':''):'Walk around ONE object: front, left, back, right. Same distance, plain background, whole object in frame. Front plus one more angle is the minimum.'}
[...$('#phgrid').children].forEach((sl,i)=>{const inp=sl.querySelector('input'),lab=sl.querySelector('label');
 inp.onchange=e=>{phLoad(i,e.target.files[0]);e.target.value=''};
 lab.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();inp.click()}};
 lab.ondragover=e=>{e.preventDefault();sl.classList.add('drag')};lab.ondragleave=()=>sl.classList.remove('drag');
 lab.ondrop=e=>{e.preventDefault();e.stopPropagation();sl.classList.remove('drag');phLoad(i,e.dataTransfer.files[0])};
 sl.querySelector('.phx').onclick=()=>{photos[i]=null;phDraw(i)}});
function loadFile(f){
 const msg=$('#upmsg');if(!f)return;
 if(!/^image\//.test(f.type)){msg.textContent='That file is not an image.';msg.className='hint bad';return}
 msg.textContent='';msg.className='hint';
 const url=URL.createObjectURL(f),im=new Image();
 im.onload=()=>{snap();blank();const s=Math.min(cv.width/im.width,cv.height/im.height),w=im.width*s,h=im.height*s;
  ctx.drawImage(im,(cv.width-w)/2,(cv.height-h)/2,w,h);URL.revokeObjectURL(url);uploaded=true;setBlank(false);if(mode!=='upload')setMode('upload');else refreshUp()};
 im.onerror=()=>{msg.textContent='Could not read that image.';msg.className='hint bad';URL.revokeObjectURL(url)};
 im.src=url}
$('#file').onchange=e=>{loadFile(e.target.files[0]);e.target.value=''};
$('#drop').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#file').click()}};
$('#upreplace').onclick=()=>$('#file').click();
$('#uptodraw').onclick=()=>setMode('draw');
$('#upremove').onclick=()=>{clearPad();refreshUp()};
const card=$('#inputcard');
card.addEventListener('dragover',e=>{e.preventDefault();card.classList.add('dragging')});
card.addEventListener('dragleave',e=>{if(!card.contains(e.relatedTarget))card.classList.remove('dragging')});
card.addEventListener('drop',e=>{e.preventDefault();card.classList.remove('dragging');if(mode!=='photos')loadFile(e.dataTransfer.files[0])});
addEventListener('paste',e=>{if(mode==='photos')return;const it=[...(e.clipboardData?e.clipboardData.items:[])].find(i=>i.type.startsWith('image/'));if(it)loadFile(it.getAsFile())});
['thinner','chunky','blue','red','green','carved wooden','metallic','clay','realistic','collectible toy'].forEach(t=>{const b=document.createElement('button');b.type='button';b.className='pill chip';b.textContent=t;b.onclick=()=>{const f=$('#fb');f.value=(f.value?f.value+', ':'')+t};$('#chips').appendChild(b)});
// ---- segmented controls (the thumb slides via --pos)
function segInit(id,cb){const el=$('#'+id),bs=[...el.querySelectorAll('button')];
 const set=b=>{bs.forEach(x=>x.classList.toggle('on',x===b));el.style.setProperty('--pos',bs.indexOf(b))};
 bs.forEach(b=>b.onclick=()=>{set(b);cb(b.dataset.v)});return v=>set(bs.find(b=>b.dataset.v===v)||bs[0])}
let speed='quality',geom='multi';
const setSpeed=segInit('speed',v=>speed=v);
setSpeed('quality');
const setGeom=segInit('geom',v=>{geom=v;const n=$('#geomnote');if(v==='multi'){n.textContent='Multi-view: Zero123++ draws the side and back views, then Hunyuan3D-2mv builds the shape. Takes a couple of minutes.';n.className='hint'}else n.className='hint hidden'});
setGeom('multi');
fetch('/health').then(r=>r.json()).then(h=>{if(h.mv&&h.mv.on===false){geom='single';const g=$('#geom');g.classList.add('hidden');g.previousElementSibling.classList.add('hidden')}
 else if(h.mv&&h.mv.default){geom='multi';setGeom('multi');const n=$('#geomnote');n.textContent='Multi-view: Zero123++ draws the side and back views, then Hunyuan3D-2mv builds the shape. Takes a couple of minutes.';n.className='hint'}}).catch(()=>{});
const PRESETS=[['original','Original'],['clay','Clay'],['terracotta','Terracotta'],['gold','Metallic gold'],['toy','Toy matte']];
function drawPresets(cur){$('#presets').innerHTML=PRESETS.map(([k,l])=>`<button type="button" class="pill ${k===cur?'on':''}" data-p="${k}">${l}</button>`).join('');
 document.querySelectorAll('#presets button').forEach(b=>b.onclick=()=>relight(b.dataset.p))}
// ---- floating cubes
function cubeHTML(sz,x,y,t,r,dl,o){return `<span class="fc" style="--s:${sz}px;--x:${x}%;--y:${y}%;--t:${t}s;--r:${r}s;--dl:${dl}s;--o:${o}"><b><i></i><i></i><i></i><i></i><i></i><i></i></b></span>`}
document.querySelectorAll('.fl').forEach(el=>{const n=+el.dataset.n||8;let h='';
 for(let k=0;k<n;k++)h+=cubeHTML(10+(k*7)%22,(+(el.dataset.x0||4)+(k*53+9)%(+(el.dataset.xs||88))),(k*31+6)%78,(3.4+(k%4)*.9).toFixed(1),(7+(k%5)*2.4).toFixed(1),-(k*.7).toFixed(1),(.45+(k%3)*.2).toFixed(2));
 el.innerHTML=h});
$('#loader').innerHTML=[0,1,2].map(k=>cubeHTML(12,0,0,1.6,5+k,-(k*.35).toFixed(2),1)).join('');
// ---- stage tiles + progress bar (built once, so their transitions can run)
const STAGES=['Read','Identify','Render','Build 3D','Check'];
const stg=$('#stepper');stg.innerHTML=STAGES.map((l,i)=>`<div class="sg"><b>0${i+1}</b>${l}</div>`).join('');
function stage(k){[...stg.children].forEach((d,i)=>{d.classList.toggle('done',i<k);d.classList.toggle('act',i===k)});$('#progbar').style.setProperty('--p',((k+.5)/STAGES.length).toFixed(3))}
function stageOf(m){m=m||'';if(/^Stage 1/.test(m))return 0;if(/^Stage 2/.test(m))return 1;if(/^Stage (3a|4)/.test(m))return 2;if(/^(Stage 3b|Loop)/.test(m))return 3;if(/^(Post|Export|Done|Tier)/.test(m))return 4;return 0}
// ---- generate
let lastInput=null,job=null,idDone=false,cDone=false,mvDone=false,candKey='',mvKey='';
function view(n){put($('#idle'),n==='idle');put($('#busy'),n==='busy');put($('#busy2'),n==='busy');put($('#res'),n==='res');put($('#err'),n==='err');if(n!=='res')put($('#top'),false)}
function steps(list){
 const box=$('#steps');
 list.forEach((s,i)=>{let d=box.children[i];if(!d){d=document.createElement('div');d.className='st';d.innerHTML='<i></i><span></span>';box.appendChild(d)}
  const sp=d.lastElementChild,h=`${esc(s.msg)}<em>${s.t}s</em>`;if(sp.dataset.h!==h){sp.innerHTML=h;sp.dataset.h=h}});
 while(box.children.length>list.length)box.lastElementChild.remove();
 [...box.children].forEach((d,i)=>d.classList.toggle('act',i===list.length-1));
 stage(list.length?stageOf(list[list.length-1].msg):0)}
function drawIdent(i){
 const alts=i.alts||[];
 const bars=alts.length?`<div class="bars">${alts.map(a=>`<button type="button" class="bar alt${a.label===i.label?' on':''}" data-l="${esc(a.label)}"><i style="--w:${Math.round(a.p*100)}"></i><span>${esc(a.label)}</span><small>${Math.round(a.p*100)}%</small></button>`).join('')}</div>`:'';
 const meta=i.view?`pose: ${esc(i.view)}`:'';
 $('#identcard').innerHTML=`<span class="lab">Identified as</span><div class="idrow"><div class="idname">${esc(i.label)}</div><span class="srcb${i.low?' low':''}">${i.low?'not sure · ':''}${esc(i.source)}</span></div>
 ${i.caption?`<p class="idcap">${esc(i.caption)}</p>`:''}
 ${meta?`<div class="idmeta">${meta}</div>`:''}
 ${i.ai_note?`<div class="warnline">The AI reader gave no answer (${esc(i.ai_note)}). The CLIP guess was used instead.</div>`:''}
 ${i.low?`<div class="warnline">Low confidence. Check the list below: if the right answer is there, click it.</div>`:''}
 ${bars}
 <div class="idhelp">${alts.length?'Click an option to use it, or type it':'Type it'} in “What is it?”, then generate again.</div>`;
 document.querySelectorAll('#identcard .alt').forEach(b=>b.onclick=()=>{const f=$('#name');f.value=b.dataset.l;f.focus();document.querySelectorAll('#identcard .alt').forEach(x=>x.classList.toggle('on',x===b))});
 put($('#top'),true)}
function drawMV(list){
 put($('#mvcard'),list.length>0);if(!list.length)return;
 const mk=list.map(v=>v.img.slice(-40)+v.used).join('|');if(mk===mvKey){put($('#top'),true);return}mvKey=mk;
 $('#mvs').innerHTML=list.map((v,i)=>`<figure style="--k:${i}" class="${v.used?'win':''}"><img src="${v.img}" alt="${esc(v.name)}" style="${v.used?'outline:2px solid var(--accent);outline-offset:3px':''}"><figcaption>${esc(v.name)}${v.used?' · used':''}</figcaption></figure>`).join('');
 put($('#top'),true)}
function drawCands(list,final){
 put($('#candcard'),list.length>0);if(!list.length)return;
 $('#candlab').textContent=final?'Generated images (winner highlighted)':'Generated images · the 3D builder gets the highlighted one';
 const key=list.map(c=>c.img.slice(-40)).join('|');
 if(key===candKey){document.querySelectorAll('#cands figure').forEach((f,i)=>{f.classList.toggle('win',!!list[i].winner);f.querySelector('.tag').textContent=final?'winner':'leading'});put($('#top'),true);return}
 candKey=key;
 $('#cands').innerHTML=list.map((c,i)=>`<figure class="${c.winner?'win':''}"><span class="tag">${final?'winner':'leading'}</span><img src="${c.img}" alt="candidate ${i+1}" style="--k:${i}"><figcaption>Candidate ${i+1} · score ${c.score}</figcaption>
 ${Object.entries(c.parts||{}).map(([k,v])=>`<div class="mt"><span>${k}</span><span>${v}</span></div><div class="bar"><i style="width:${Math.round(v*100)}%"></i></div>`).join('')}</figure>`).join('');
 document.querySelectorAll('#cands img').forEach(im=>{const on=()=>requestAnimationFrame(()=>im.classList.add('in'));im.complete?on():im.addEventListener('load',on,{once:true})});
 put($('#top'),true)}
async function generate(){
 if(mode==='photos'&&(!photos[0]||photos.filter(Boolean).length<2)){$('#err').textContent='Add a front photo and at least one more angle first.';view('err');return}
 const rid=Math.random().toString(36).slice(2,14);lastInput=mode==='photos'?photos[0]:cv.toDataURL('image/jpeg',.8);$('#go').disabled=true;idDone=cDone=mvDone=false;candKey='';mvKey='';drawMV([]);
 $('.stage').classList.remove('fin');$('#steps').innerHTML='';view('busy');stage(0);steps([{msg:'Sending',t:0}]);$('#wellmsg').textContent='Working on it…';
 const poll=setInterval(async()=>{try{const r=await(await fetch('/api/progress/'+rid)).json();if(r.steps.length)steps(r.steps);const p=r.partial||{};
  if(p.identified&&!idDone){idDone=true;drawIdent(p.identified)}
  if(p.mv_views&&!mvDone){mvDone=true;drawMV(p.mv_views);$('#wellmsg').textContent='Building the 3D model from the views…'}
  if(p.candidates&&!cDone){cDone=true;drawCands(p.candidates,false);$('#wellmsg').textContent='Building the 3D model from the leading image…'}}catch(e){}},600);
 try{
  const isP=mode==='photos';
  const r=await fetch(isP?'/api/photos3d':'/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(isP?{images:photos,name:$('#name').value,feedback:$('#fb').value,quality:speed,rid}:{image:cv.toDataURL('image/png'),name:$('#name').value,feedback:$('#fb').value,mode:$('#mode').value,cpu:$('#cpu').checked,quality:speed,multiview:geom==="multi",rid})});
  const t0=await r.text();let j0;try{j0=JSON.parse(t0)}catch(e){throw new Error('The server sent a page instead of data (status '+r.status+'): '+t0.slice(0,120))}
  if(!r.ok)throw new Error(j0.error||r.statusText);
  let j=null;
  while(!j){await new Promise(s=>setTimeout(s,1500));let t=null;
   try{const rr=await fetch('/api/result/'+rid);t=await rr.json()}catch(e){continue}
   if(t.status==='done')j=t.result;else if(t.status==='error')throw new Error(t.error)}
  await show(j);
 }catch(e){$('#err').textContent=e.message;view('err')}
 finally{clearInterval(poll);$('#go').disabled=false}
}
$('#go').onclick=generate;
function dls(d){$('#dl').innerHTML=Object.entries(d).map(([k,v])=>`<a href="${v}" download>Download .${k}</a>`).join('')}
function vws(v){$('#views').innerHTML=v.map((x,i)=>`<figure style="--k:${i}"><img src="${x.img}" alt="${esc(x.name)} view"><figcaption>${esc(x.name)}</figcaption></figure>`).join('')}
function stat(big,label,sub,d,hero){return `<div class="stat${hero?' hero':''}" style="--d:${d}ms"><div class="big">${big}</div><div class="slab">${esc(label)}</div><div class="ssub">${esc(sub)}</div></div>`}
const num=(n,sup,d)=>`<span class="num" style="--to:${Math.round(n)};--d:${d||0}ms"></span>${sup?`<sup>${sup}</sup>`:''}`;
async function show(j){
 // 1) fade the working screen out, 2) put the summary + 3D model on top and fade them in, 3) the identified / images cards settle in below
 idDone=cDone=mvDone=true;const st=$('.stage');
 ['#top','#busy','#busy2','#err','#idle'].forEach(q=>put($(q),false));
 await new Promise(r=>setTimeout(r,280));
 st.classList.add('fin');job=j.job;view('res');
 if(j.identified)drawIdent(j.identified);drawCands(j.candidates||[],true);drawMV(j.mv_views||[]);
 const m=$('#mv');m.classList.remove('ready');m.addEventListener('load',()=>m.classList.add('ready'),{once:true});setTimeout(()=>m.classList.add('ready'),6000);
 m.src=j.glb;dls(j.downloads);vws(j.views);drawPresets(j.preset||'original');
 const r=j.report,pct=Math.round(r.sil_iou*100),good=r.sil_iou>=r.iou_retry;


 $('#chip').className='chipv';$('#chip').textContent='Model ready';
 const lvl=pct>=70?3:pct>=45?2:1,lvlTxt=['','Creative take','Follows your input','Closely follows'][lvl];
 const iso=(j.views&&j.views[3])?j.views[3].img:'';
 const srcWord=j.kind==='photos'?'photos':(j.kind==='sketch'?'sketch':'image');
 const IC={tri:'<path d="M12 3 21 19H3z"/>',time:'<circle cx="12" cy="13" r="8"/><path d="M12 9v4l3 2M9 2h6"/>',print:'<path d="M6 9V3h12v6M6 18H4a1 1 0 0 1-1-1v-6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v6a1 1 0 0 1-1 1h-2"/><rect x="6" y="14" width="12" height="7" rx="1"/>',fit:'<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r=".8" fill="currentColor"/>'};
 const tile=(ic,big,label,sub,d)=>`<div class="stat tile" style="--d:${d}ms"><span class="ti"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${IC[ic]}</svg></span><div class="big">${big}</div><div class="slab">${esc(label)}</div><div class="ssub">${sub}</div></div>`;
 let fl='';for(let k=0;k<7;k++)fl+=cubeHTML(12+(k*9)%20,(k*47+8)%92,(k*29+4)%70,(3.4+(k%4)*.9).toFixed(1),(7+(k%5)*2.4).toFixed(1),-(k*.7).toFixed(1),.5);
 $('#stats').innerHTML=`<div class="hero2"><div class="fl">${fl}</div>
  <div class="h2txt"><span class="h2eye">Built from your ${srcWord}</span><div class="h2name">${esc(j.label)}</div>
   <div class="h2chips"><span class="h2chip">&#10003; 3D model ready</span><span class="h2chip">${esc(lvlTxt)}</span>${r.watertight?'<span class="h2chip">Ready to print</span>':''}${j.feedback&&j.feedback.hits.length?'<span class="h2chip">'+esc(j.feedback.hits[0])+'</span>':''}</div></div>
  <div class="h2art">${lastInput?`<figure class="h2in"><img src="${lastInput}" alt="your input"><figcaption>Your ${srcWord}</figcaption></figure>`:''}
   <svg class="h2arrow" viewBox="0 0 64 24" width="64" height="24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h54M46 3l10 9-10 9"/></svg>
   ${iso?`<figure class="h2out"><img src="${iso}" alt="3D result"><figcaption>3D result</figcaption></figure>`:''}</div></div>`
  +tile('tri',num(r.faces/1000,'k',250),'Triangles','mesh detail',250)
  +tile('time',num(j.total,'s',350),'Total time','start to finish',350)
  +tile('print',`<span class="bigt">${r.watertight?'Yes':'Check'}</span>`,'Print-ready',r.watertight?'watertight mesh':'may need repair in a slicer',450)
  +tile('fit',`<span class="meter">${[1,2,3].map(i=>`<i class="${i<=lvl?'on':''}" style="--i:${i}"></i>`).join('')}</span>`,'Fidelity',esc(lvlTxt),550);
 $('#kpis').innerHTML=`<div class="kpi"><b>${pct}%</b><span>Shape overlap</span></div>
 <div class="kpi"><b class="${r.watertight?'ok':'bad'}">${r.watertight?'Yes':'No'}</b><span>Watertight</span></div>
 <div class="kpi"><b>${r.faces.toLocaleString()}</b><span>Faces</span></div>
 <div class="kpi"><b>${j.total}s</b><span>End to end</span></div>
 <div class="kpi"><b style="font-size:15px">${r.extents.join(' × ')}</b><span>Bounding box</span></div>
 <div class="kpi"><b>${r.volume??'–'}</b><span>Volume (unit³)</span></div>
 <div class="kpi"><b class="${r.reload_ok?'ok':'bad'}">${r.reload_ok?'Pass':'Fail'}</b><span>.glb reload check</span></div>
 <div class="kpi"><b>×${r.z_boost}</b><span>Depth correction</span></div>`+Object.entries(r.per_view||{}).map(([k,v])=>`<div class="kpi"><b>${Math.round(v*100)}%</b><span>${k} photo overlap</span></div>`).join('');
 let n=`<div class="note"><b>${esc(j.tier)}</b> · model: ${esc(r.backend)} · object: ${esc(j.label)} · input: ${esc(j.kind)}`;
 if(r.attempts.length)n+=` · attempts: ${r.attempts.map(a=>'cand '+a.candidate+' → '+a.iou).join(', ')}`;
 if(j.feedback.hits.length)n+=` · feedback: ${esc(j.feedback.hits.join(', '))}`;
 n+='</div>'+j.notes.map(x=>`<div class="note">${esc(x)}</div>`).join('');$('#notes').innerHTML=n;
 const T=Object.entries(j.timings),mx=Math.max(...T.map(x=>x[1]),0.01);
 $('#tim').innerHTML=T.map(([k,v],i)=>`<div class="tm"><span>${esc(k)}</span><div class="tb"><i style="width:${Math.max(2,Math.round(v/mx*100))}%;animation-delay:${300+i*60}ms"></i></div><span>${v}s</span></div>`).join('');
 document.querySelectorAll('#kpis .kpi,#dl a,#presets .pill').forEach((e,i)=>e.style.setProperty('--k',i));
 setTimeout(()=>{const r=$('#res').getBoundingClientRect();if(r.top<-20||r.top>innerHeight*.6)$('#res').scrollIntoView({behavior:'smooth',block:'start'})},450);
}
async function relight(p){
 if(!job)return;drawPresets(p);
 const r=await fetch('/api/relight',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job,preset:p})});
 const j=await r.json();if(!r.ok)return;$('#mv').src=j.glb;dls(j.downloads);vws(j.views)}
</script></body></html>
"""


# ============================================================ launcher (Kaggle / local)
def _start_tunnel(port):
    exe = os.path.join(BASE, "cloudflared")
    if not os.path.exists(exe):
        urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", exe)
        os.chmod(exe, 0o755)
    p = subprocess.Popen([exe, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
        if m:
            print("\n" + "=" * 60 + f"\n  PUBLIC URL: {m.group(0)}\n" + "=" * 60 + "\n", flush=True)
            break
    for _ in p.stdout:
        pass


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--tunnel", action="store_true", help="open a public trycloudflare.com URL (for Kaggle)")
    args = ap.parse_args()
    if torch is not None and torch.cuda.is_available():
        _free, _total = torch.cuda.mem_get_info()
        print(f"[gpu] {_free / 2**30:.1f} GiB free of {_total / 2**30:.1f} GiB before loading")
        if _free < 0.9 * _total:
            print("[gpu] WARNING: GPU is already partly in use by another process. Restart the session and run only this cell.")
    load_models()
    print("[health]", {k: S[k] is not None for k in ("sd", "clip", "rembg", "tsr")}, S["err"])
    if args.tunnel:
        threading.Thread(target=_start_tunnel, args=(args.port,), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")