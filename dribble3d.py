"""Modal deploy for dribble3d. Put this next to dribble3d.py, then:  modal deploy modal_app.py

TRELLIS.2 (Microsoft) is the third geometry option. It needs its own image (CUDA 12.4 toolkit + torch 2.6, compiled extensions), so it runs as a
second container (class Trellis) that the main app calls. One-time setup for it:
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
where the token belongs to a Hugging Face account that has accepted the licences of  facebook/dinov3-vitl16-pretrain-lvd1689m  and  briaai/RMBG-2.0
(both are gated; TRELLIS.2 loads them). The first deploy compiles several CUDA packages and downloads ~15 GB: expect a long build, once.
To deploy without TRELLIS.2 (the button is then greyed out):   D3_TRELLIS=0 modal deploy modal_app.py
"""
import os

import modal

APP_DIR = "/root/app"
TSR_DIR = "/root/TripoSR"
HY_DIR = "/root/Hunyuan3D-2"
TRELLIS_DIR = "/root/TRELLIS.2"
ENABLE_TRELLIS = os.environ.get("D3_TRELLIS", "1") != "0"


def prefetch():
    """Runs once at image build (CPU, no GPU billing): bakes the weights into the image so cold starts don't download."""
    from huggingface_hub import snapshot_download
    snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
                      allow_patterns=["*.json", "*.txt", "unet/diffusion_pytorch_model.safetensors",
                                      "vae/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors"])
    snapshot_download("lllyasviel/control_v11p_sd15_scribble", allow_patterns=["*.json", "diffusion_pytorch_model.safetensors"])
    snapshot_download("openai/clip-vit-large-patch14", allow_patterns=["*.json", "*.txt", "model.safetensors"])
    snapshot_download("stabilityai/TripoSR")
    snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", allow_patterns=["*.json", "*.txt", "*.safetensors", "merges.txt", "vocab.json"])
    # Photoreal 2D stage: SDXL + scribble ControlNet + fp16-safe VAE. A failure here only means the weights download on first use.
    try:
        snapshot_download("stabilityai/stable-diffusion-xl-base-1.0",
                          allow_patterns=["*.json", "*.txt", "unet/diffusion_pytorch_model.fp16.safetensors",
                                          "vae/diffusion_pytorch_model.fp16.safetensors", "text_encoder/model.fp16.safetensors",
                                          "text_encoder_2/model.fp16.safetensors"])
        snapshot_download("xinsir/controlnet-scribble-sdxl-1.0", allow_patterns=["*.json", "diffusion_pytorch_model.safetensors"])
        snapshot_download("madebyollin/sdxl-vae-fp16-fix", allow_patterns=["*.json", "diffusion_pytorch_model.safetensors"])
    except Exception as e:
        print("[prefetch sdxl skipped]", repr(e)[:300])
    from rembg import new_session
    new_session("u2net")                      # fallback matting model
    try:
        new_session("isnet-general-use")      # default background remover: cleaner object edges and thin parts than u2net
    except Exception as e:
        print("[prefetch isnet skipped]", repr(e)[:200])
    # Multi-view + 4 Photos: Zero123++ (one image -> 6 views) and Hunyuan3D-2mv (views -> mesh), about 15 GB together
    try:   # a failure here must not block the whole build; the models would then download on first use instead
        import sys
        import torch
        from diffusers import DiffusionPipeline
        DiffusionPipeline.from_pretrained("sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
                                          trust_remote_code=True, torch_dtype=torch.float16)
        sys.path.insert(0, HY_DIR)
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv",
                                                         use_safetensors=True, variant="fp16", device="cpu")

    except Exception as e:
        print("[prefetch multi-view skipped]", repr(e)[:300])


def prefetch_trellis():
    """Runs once at image build (CPU): loading the pipeline once bakes TRELLIS.2-4B, DINOv3, RMBG-2.0 and the sparse-structure decoder into the image."""
    import sys
    sys.path.insert(0, TRELLIS_DIR)
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    try:   # a failure here only means the weights download on the first TRELLIS.2 run instead
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    except Exception as e:
        print("[prefetch trellis2 skipped]", repr(e)[:400])


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libglu1-mesa", "libopengl0", "libxrender1")
    .pip_install("torch", "torchvision", "diffusers", "huggingface_hub", "accelerate", "safetensors", "omegaconf", "einops",
                 "trimesh", "networkx", "rembg", "onnxruntime", "imageio", "fast-simplification", "scikit-image", "peft",
                 "fastapi", "uvicorn", "pillow", "scipy", "pyyaml", "tqdm", "opencv-python-headless", "numpy", "pymeshlab")
    .pip_install("transformers==5.8.1")          # same pin as the notebook: TripoSR checkpoints break on 5.9+
    .env({"HF_HOME": "/root/hf", "U2NET_HOME": "/root/.u2net", "TSR_DIR": TSR_DIR, "HY_DIR": HY_DIR,
          "D3_OUT": "/tmp/d3_out", "D3_MV": "1", "D3_MV_DEFAULT": "1", "D3_VLM_ID": "Qwen/Qwen2.5-VL-7B-Instruct",
          "D3_HYMV_RES": "448", "D3_HYMV_STEPS": "60", "D3_MV_STEPS": "100",
          "D3_SD": "xl", "D3_MV_TRIES": "3", "D3_FACES": "400000", "D3_SMOOTH": "3",
          "D3_TRELLIS": "1" if ENABLE_TRELLIS else "0"})   # bigger GPU: finer Hunyuan3D mesh + more steps
    .run_commands(f"git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git {TSR_DIR}",
                  f"git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git {HY_DIR}")
    .run_function(prefetch)
    .add_local_file("dribble3d.py", f"{APP_DIR}/dribble3d.py")
)

app = modal.App("dribble3d")

if ENABLE_TRELLIS:
    hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
    # Same steps as the official setup.sh (CUDA 12.4, torch 2.6.0), run without conda. sm_89 = the L40S.
    trellis_image = (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
        .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libjpeg-dev", "libxrender1", "libopengl0", "libglu1-mesa")
        .pip_install("pip", "wheel", "setuptools", "packaging", "ninja", "psutil")
        .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
        .env({"CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -pthread -shared", "LDCXXSHARED": "g++ -pthread -shared",   # the bundled Python's sysconfig says clang++, which this image lacks
              "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "4", "HF_HOME": "/root/hf",
              "OPENCV_IO_ENABLE_OPENEXR": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        .run_commands(
            f"git clone -b main --recursive https://github.com/microsoft/TRELLIS.2.git {TRELLIS_DIR}",
            f"cd {TRELLIS_DIR} && bash ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm",
            gpu="L40S",
        )
        .run_function(prefetch_trellis, secrets=[hf_secret], cpu=4, memory=32768, timeout=3600)
    )

    # Own container because TRELLIS.2 needs >= 24 GB VRAM and a different torch than the main app. Takes a PNG cut-out, returns GLB bytes.
    @app.cls(image=trellis_image, gpu="L40S", cpu=4, memory=49152, max_containers=1, scaledown_window=1200, timeout=1800, secrets=[hf_secret])
    class Trellis:
        @modal.enter()
        def load(self):
            import sys
            sys.path.insert(0, TRELLIS_DIR)
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            self.pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
            self.pipe.cuda()

        @modal.method()
        def build(self, png: bytes, seed: int = 1, faces: int = 400000, tex: int = 2048) -> bytes:
            import io
            import o_voxel
            from PIL import Image
            img = Image.open(io.BytesIO(png))
            img.load()
            try:
                mesh = self.pipe.run(img, seed=int(seed))[0]
            except TypeError:                                  # older/newer signature without a seed argument
                mesh = self.pipe.run(img)[0]
            mesh.simplify(16777216)                            # nvdiffrast limit
            glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                                             attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                             decimation_target=int(faces), texture_size=int(tex), remesh=True, remesh_band=1, remesh_project=0,
                                             verbose=False)
            return glb.export(file_type="glb")


# max_containers=1 is required: job progress/results live in the process's memory, so every request must hit the same container.
@app.cls(image=image, gpu="L40S", cpu=4, memory=49152, max_containers=1, scaledown_window=1200, timeout=3600)
@modal.concurrent(max_inputs=32)                 # the page polls /api/progress while the GPU thread is busy
class Dribble:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, APP_DIR)
        import dribble3d as d
        d.load_models()
        if ENABLE_TRELLIS:                                # third geometry option: hand the cut-out to the TRELLIS.2 container
            d.TRELLIS_REMOTE = lambda png, seed: Trellis().build.remote(png, seed, d.TR_FACES, d.TR_TEX)
        print("load errors:", d.S["err"])
        self.d = d

    @modal.asgi_app()
    def web(self):
        return self.d.app