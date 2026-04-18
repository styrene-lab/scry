#!/usr/bin/env python3
"""
Scry diffusers worker — long-running sidecar process.

Reads JSON requests from stdin (one per line), executes diffusers pipelines,
writes JSON responses to stdout (one per line). Stderr is for logging.

Protocol:
  Request:  {"id": "...", "method": "generate|img2img|upscale|health|shutdown", "params": {...}}
  Response: {"id": "...", "result": {...}} or {"id": "...", "error": {"code": "...", "message": "..."}}
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def log(msg: str) -> None:
    print(f"[scry-worker] {msg}", file=sys.stderr, flush=True)


# Fail fast with a clear message if deps are missing.
try:
    import torch
except ImportError as e:
    log(f"FATAL: {e}")
    log(f"Python: {sys.executable} ({sys.version})")
    log("torch is not installed. Run: pip install torch")
    sys.exit(1)

try:
    import diffusers  # noqa: F401
except ImportError as e:
    log(f"FATAL: {e}")
    log("diffusers is not installed. Run: pip install diffusers transformers accelerate")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------

class PipelineCache:
    """Holds the currently loaded pipeline to avoid reloading on every call."""

    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: str | None = None
        self.pipeline_type: str | None = None
        self.loras_loaded: list[str] = []
        self.device: str = self._pick_device()
        self.dtype: torch.dtype = self._pick_dtype()
        log(f"device={self.device} dtype={self.dtype}")

    def _pick_dtype(self) -> torch.dtype:
        if self.device == "cpu":
            return torch.float32
        if self.device == "mps":
            # MPS has inconsistent bfloat16 support; float16 is reliable.
            return torch.float16
        return torch.bfloat16

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self, model_path: str, pipeline_type: str) -> Any:
        if self.pipe and self.model_path == model_path and self.pipeline_type == pipeline_type:
            return self.pipe

        self._unload()
        log(f"loading {pipeline_type} pipeline: {model_path}")

        if pipeline_type == "txt2img":
            from diffusers import AutoPipelineForText2Image as Cls
        else:
            from diffusers import AutoPipelineForImage2Image as Cls

        pipe = Cls.from_pretrained(model_path, torch_dtype=self.dtype)

        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)

        self.pipe = pipe
        self.model_path = model_path
        self.pipeline_type = pipeline_type
        self.loras_loaded = []
        log(f"{pipeline_type} pipeline ready")
        return pipe

    def apply_loras(self, pipe: Any, loras: list[dict]) -> Any:
        wanted = [l["path"] for l in loras]
        if self.loras_loaded == wanted:
            return pipe

        if self.loras_loaded:
            try:
                pipe.unfuse_lora()
                pipe.unload_lora_weights()
            except Exception:
                pass
            self.loras_loaded = []

        if not loras:
            return pipe

        adapter_names = []
        adapter_weights = []
        for i, lora in enumerate(loras):
            name = f"lora_{i}"
            log(f"loading LoRA: {lora['path']} (weight={lora.get('weight', 0.8)})")
            pipe.load_lora_weights(lora["path"], adapter_name=name)
            adapter_names.append(name)
            adapter_weights.append(lora.get("weight", 0.8))

        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        pipe.fuse_lora()
        self.loras_loaded = wanted
        return pipe

    def _unload(self) -> None:
        if self.pipe is not None:
            log("unloading current pipeline")
            del self.pipe
            self.pipe = None
            self.model_path = None
            self.pipeline_type = None
            self.loras_loaded = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


cache = PipelineCache()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def resolve_seed(params: dict) -> tuple[int, torch.Generator]:
    """Resolve seed and return (seed, generator).

    Generator is always on CPU — MPS doesn't reliably support torch.Generator,
    and diffusers pipelines handle CPU→device transfer internally.
    """
    seed = params.get("seed", -1)
    if seed < 0:
        seed = torch.randint(0, 2**32, (1,), device="cpu").item()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return seed, generator


def save_images(
    images: list,
    output_dir: str,
    seed: int,
    prefix: str,
) -> list[str]:
    """Save PIL images to disk and return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    timestamp = int(time.time())
    for i, img in enumerate(images):
        filename = f"scry_{prefix}_{timestamp}_{seed}_{i}.png"
        path = out / filename
        img.save(path)
        paths.append(str(path))
        log(f"saved: {path}")
    return paths


def build_gen_kwargs(params: dict, seed: int, generator: torch.Generator) -> dict[str, Any]:
    """Build the common kwargs dict for diffusers pipeline calls."""
    kwargs: dict[str, Any] = {
        "prompt": params["prompt"],
        "num_inference_steps": params.get("steps", 20),
        "guidance_scale": params.get("cfg_scale", 7.0),
        "generator": generator,
    }
    negative = params.get("negative_prompt", "")
    if negative:
        kwargs["negative_prompt"] = negative
    return kwargs


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_health(_params: dict) -> dict:
    return {
        "available": True,
        "name": "diffusers",
        "device": cache.device,
        "dtype": str(cache.dtype),
        "torch_version": torch.__version__,
    }


def handle_generate(params: dict) -> dict:
    pipe = cache.load(params["model_path"], "txt2img")
    pipe = cache.apply_loras(pipe, params.get("loras", []))

    seed, generator = resolve_seed(params)
    kwargs = build_gen_kwargs(params, seed, generator)
    kwargs["width"] = params.get("width", 1024)
    kwargs["height"] = params.get("height", 1024)
    kwargs["num_images_per_prompt"] = params.get("batch_size", 1)

    log(f"generating: steps={kwargs['num_inference_steps']} size={kwargs['width']}x{kwargs['height']} seed={seed}")
    t0 = time.monotonic()
    result = pipe(**kwargs)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    log(f"generation complete in {elapsed_ms}ms")

    paths = save_images(result.images, params.get("output_dir", "."), seed, "gen")
    return {"images": paths, "seed": seed, "elapsed_ms": elapsed_ms, "model": params["model_path"]}


def handle_img2img(params: dict) -> dict:
    from diffusers.utils import load_image

    pipe = cache.load(params["model_path"], "img2img")
    pipe = cache.apply_loras(pipe, params.get("loras", []))

    seed, generator = resolve_seed(params)
    kwargs = build_gen_kwargs(params, seed, generator)
    kwargs["image"] = load_image(params["input_image"])
    kwargs["strength"] = params.get("strength", 0.7)
    kwargs["num_images_per_prompt"] = params.get("batch_size", 1)

    log(f"img2img: strength={kwargs['strength']} seed={seed}")
    t0 = time.monotonic()
    result = pipe(**kwargs)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    paths = save_images(result.images, params.get("output_dir", "."), seed, "i2i")
    return {"images": paths, "seed": seed, "elapsed_ms": elapsed_ms, "model": params["model_path"]}


def handle_upscale(params: dict) -> dict:
    from diffusers.utils import load_image

    input_image = load_image(params["input_image"])
    factor = params.get("factor", 2)

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import numpy as np

        upscaler_path = params.get("upscaler_path")
        if not upscaler_path or not Path(upscaler_path).exists():
            raise ImportError("no upscaler model path")

        log(f"upscaling with RealESRGAN: factor={factor}")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=factor)
        upsampler = RealESRGANer(scale=factor, model_path=upscaler_path, model=model, half=True)

        img_array = np.array(input_image)
        t0 = time.monotonic()
        output, _ = upsampler.enhance(img_array, outscale=factor)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        from PIL import Image
        output_img = Image.fromarray(output)

    except (ImportError, Exception) as e:
        log(f"RealESRGAN not available ({e}), falling back to Lanczos resize")
        from PIL import Image
        w, h = input_image.size
        t0 = time.monotonic()
        output_img = input_image.resize((w * factor, h * factor), Image.LANCZOS)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

    paths = save_images([output_img], params.get("output_dir", "."), 0, f"up{factor}x")
    return {"images": paths, "seed": 0, "elapsed_ms": elapsed_ms, "model": "upscaler"}


# ---------------------------------------------------------------------------
# Model search/download
# ---------------------------------------------------------------------------

def handle_hf_search(params: dict) -> dict:
    """Search HuggingFace Hub for models."""
    from huggingface_hub import HfApi

    api = HfApi()
    query = params.get("query", "")
    model_type = params.get("filter", "")  # e.g. "text-to-image"
    limit = params.get("limit", 20)

    kwargs: dict[str, Any] = {"search": query, "limit": limit, "sort": "downloads"}
    if model_type:
        kwargs["pipeline_tag"] = model_type

    models = api.list_models(**kwargs)

    results = []
    for m in models:
        results.append({
            "id": m.id,
            "downloads": getattr(m, "downloads", 0),
            "likes": getattr(m, "likes", 0),
            "pipeline_tag": getattr(m, "pipeline_tag", None),
            "tags": list(getattr(m, "tags", []))[:10],
            "last_modified": str(getattr(m, "last_modified", "")),
        })

    return {"query": query, "count": len(results), "models": results}


def handle_hf_download(params: dict) -> dict:
    """Download a model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download, snapshot_download

    repo_id = params["repo_id"]
    filename = params.get("filename")
    dest_dir = params.get("dest_dir")

    log(f"downloading from HF: {repo_id}" + (f" file={filename}" if filename else " (full snapshot)"))

    if filename:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_dir,
        )
        return {"downloaded": True, "path": str(path), "repo_id": repo_id, "filename": filename}
    else:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=dest_dir,
        )
        return {"downloaded": True, "path": str(path), "repo_id": repo_id}


def handle_civitai_search(params: dict) -> dict:
    """Search CivitAI for models."""
    import urllib.request
    import urllib.parse

    query = params.get("query", "")
    model_type = params.get("type", "")  # Checkpoint, LORA, ControlNet, etc.
    limit = params.get("limit", 20)
    sort = params.get("sort", "Highest Rated")

    url_params: dict[str, str] = {"limit": str(limit), "sort": sort}
    if query:
        url_params["query"] = query
    if model_type:
        url_params["types"] = model_type

    url = f"https://civitai.com/api/v1/models?{urllib.parse.urlencode(url_params)}"

    try:
        req = urllib.request.Request(url, headers={
            "Content-Type": "application/json",
            "User-Agent": "scry/0.1.0",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "query": query, "count": 0, "models": []}

    results = []
    for m in data.get("items", []):
        latest = m.get("modelVersions", [{}])[0] if m.get("modelVersions") else {}
        results.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "type": m.get("type"),
            "nsfw": m.get("nsfw", False),
            "tags": m.get("tags", [])[:10],
            "stats": {
                "downloads": m.get("stats", {}).get("downloadCount", 0),
                "rating": m.get("stats", {}).get("rating", 0),
                "favorites": m.get("stats", {}).get("favoriteCount", 0),
            },
            "latest_version": {
                "id": latest.get("id"),
                "name": latest.get("name"),
                "base_model": latest.get("baseModel"),
            },
        })

    return {"query": query, "count": len(results), "models": results}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

HANDLERS = {
    "health": handle_health,
    "generate": handle_generate,
    "img2img": handle_img2img,
    "upscale": handle_upscale,
    "hf_search": handle_hf_search,
    "hf_download": handle_hf_download,
    "civitai_search": handle_civitai_search,
}


def main() -> None:
    log("worker started, waiting for requests")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"id": None, "error": {"code": "ParseError", "message": str(e)}}), flush=True)
            continue

        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "shutdown":
            log("shutdown requested")
            print(json.dumps({"id": req_id, "result": {"shutdown": True}}), flush=True)
            break

        handler = HANDLERS.get(method)
        if handler is None:
            print(json.dumps({"id": req_id, "error": {"code": "MethodNotFound", "message": f"unknown method: {method}"}}), flush=True)
            continue

        try:
            result = handler(params)
            response = {"id": req_id, "result": result}
        except Exception as e:
            log(f"error in {method}: {traceback.format_exc()}")
            response = {"id": req_id, "error": {"code": "InternalError", "message": str(e)}}

        print(json.dumps(response), flush=True)

    log("worker exiting")


if __name__ == "__main__":
    main()
