"""Modal deploy for dribble3d. Put this next to dribble3d.py, then:  modal deploy modal_app.py"""
import modal

APP_DIR = "/root/app"
TSR_DIR = "/root/TripoSR"
HY_DIR = "/root/Hunyuan3D-2"


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
          "D3_SD": "xl", "D3_FACES": "400000", "D3_SMOOTH": "3"})   # bigger GPU: finer Hunyuan3D mesh + more steps
    .run_commands(f"git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git {TSR_DIR}",
                  f"git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git {HY_DIR}")
    .run_function(prefetch)
    .add_local_file("dribble3d.py", f"{APP_DIR}/dribble3d.py")
)

app = modal.App("dribble3d")


# max_containers=1 is required: job progress/results live in the process's memory, so every request must hit the same container.
@app.cls(image=image, gpu="L40S", cpu=4, memory=49152, max_containers=1, scaledown_window=300, timeout=3600)
@modal.concurrent(max_inputs=32)                 # the page polls /api/progress while the GPU thread is busy
class Dribble:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, APP_DIR)
        import dribble3d as d
        d.load_models()
        if d.MV_ON:                                   # preload multi-view models (parked in CPU RAM) so the first run isn't slow
            for key, fn in (("zpp", d._load_zpp), ("hymv", d._load_hymv)):
                try:
                    d.S[key] = fn()
                except Exception as e:
                    d.S["err"][key] = repr(e)[:300]
        print("load errors:", d.S["err"])
        self.d = d

    @modal.asgi_app()
    def web(self):
        return self.d.app
