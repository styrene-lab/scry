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
import os
import sys
import threading
import time
import traceback
import uuid
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


class DownloadJob:
    """One in-flight or completed HF download.

    State is mutated from the download thread and read from the main (RPC) thread.
    Atomic field assignments are sufficient — no compound updates.
    """

    def __init__(
        self,
        job_id: str,
        repo_id: str,
        filename: str | None,
        dest_dir: str,
        log_path: str,
    ) -> None:
        self.job_id = job_id
        self.repo_id = repo_id
        self.filename = filename
        self.dest_dir = dest_dir
        self.log_path = log_path
        self.state = "pending"  # pending | downloading | complete | failed
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.bytes_total = 0
        self.files_total = 0
        self.bytes_done = 0
        self.result_path: str | None = None
        self.error: str | None = None

    def snapshot(self) -> dict:
        return {
            "job_id": self.job_id,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "state": self.state,
            "log_path": self.log_path,
            "dest_dir": self.dest_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "bytes_total": self.bytes_total,
            "bytes_done": self.bytes_done,
            "files_total": self.files_total,
            "result_path": self.result_path,
            "error": self.error,
            "elapsed_s": (self.finished_at or time.time()) - self.started_at,
        }


_JOBS: dict[str, DownloadJob] = {}
_JOBS_LOCK = threading.Lock()


def _walk_dir_size(p: Path) -> int:
    """Sum the size of every regular file under p. Skips symlinks (HF cache uses
    symlinks from snapshots/ -> blobs/, so following them would double-count).
    """
    if not p.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(p, followlinks=False):
        for name in files:
            fp = Path(root) / name
            try:
                if not fp.is_symlink():
                    total += fp.stat().st_size
            except OSError:
                pass
    return total


def _hf_cache_dir_for_repo(repo_id: str) -> Path:
    """Where snapshot_download lands files when local_dir is None."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"


def _probe_repo_size(repo_id: str, filename: str | None) -> tuple[int, int]:
    """Best-effort: return (bytes_total, files_total). Zero if probe fails."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id, files_metadata=True)
        siblings = info.siblings or []
        if filename:
            for s in siblings:
                if s.rfilename == filename:
                    return (s.size or 0, 1)
            return (0, 1)
        return (sum((s.size or 0) for s in siblings), len(siblings))
    except Exception as e:  # noqa: BLE001 — probe is opportunistic
        log(f"could not probe repo size for {repo_id}: {e}")
        return (0, 0)


def _run_download(job: DownloadJob) -> None:
    """Download body — executed on a dedicated thread."""
    from huggingface_hub import hf_hub_download, snapshot_download

    watch_dir = Path(job.dest_dir)
    stop_watcher = threading.Event()

    def watcher() -> None:
        # Open in append mode so the header written below stays at the top.
        with open(job.log_path, "a", buffering=1) as logf:
            while not stop_watcher.wait(2.0):
                done = _walk_dir_size(watch_dir)
                job.bytes_done = done
                if job.bytes_total > 0:
                    pct = 100.0 * done / job.bytes_total
                    logf.write(
                        f"[{time.strftime('%H:%M:%S')}] {done:,} / {job.bytes_total:,} bytes ({pct:.1f}%)\n"
                    )
                else:
                    logf.write(f"[{time.strftime('%H:%M:%S')}] {done:,} bytes\n")

    watcher_thread = threading.Thread(
        target=watcher, daemon=True, name=f"scry-watch-{job.job_id}"
    )
    watcher_thread.start()

    try:
        job.state = "downloading"
        local_dir_arg = job.dest_dir if Path(job.dest_dir) != _hf_cache_dir_for_repo(job.repo_id) else None
        if job.filename:
            path = hf_hub_download(
                repo_id=job.repo_id,
                filename=job.filename,
                local_dir=local_dir_arg,
            )
        else:
            path = snapshot_download(
                repo_id=job.repo_id,
                local_dir=local_dir_arg,
            )
        job.result_path = str(path)
        job.bytes_done = _walk_dir_size(watch_dir)
        job.state = "complete"
    except Exception as e:  # noqa: BLE001 — recorded on the job
        job.error = f"{type(e).__name__}: {e}"
        job.state = "failed"
        with open(job.log_path, "a") as logf:
            logf.write(f"\nERROR: {traceback.format_exc()}\n")
    finally:
        stop_watcher.set()
        watcher_thread.join(timeout=3.0)
        job.finished_at = time.time()
        with open(job.log_path, "a") as logf:
            logf.write(
                f"\n# finished {time.strftime('%Y-%m-%d %H:%M:%S')} state={job.state}"
            )
            if job.result_path:
                logf.write(f" path={job.result_path}")
            if job.error:
                logf.write(f" error={job.error}")
            logf.write("\n")


def handle_hf_download(params: dict) -> dict:
    """Start a HuggingFace download in a background thread; return a job handle.

    The download runs asynchronously — the caller polls `job_status` for progress
    or tails `log_path` directly.
    """
    repo_id = params["repo_id"]
    filename = params.get("filename")
    dest_dir = params.get("dest_dir")

    if dest_dir:
        watch_dir = Path(dest_dir).expanduser()
    else:
        watch_dir = _hf_cache_dir_for_repo(repo_id)

    job_id = uuid.uuid4().hex[:8]
    log_dir = Path("~/.cache/scry/downloads").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"

    bytes_total, files_total = _probe_repo_size(repo_id, filename)

    with open(log_path, "w", buffering=1) as logf:
        logf.write(f"# scry download job {job_id}\n")
        logf.write(f"# repo: {repo_id}\n")
        if filename:
            logf.write(f"# file: {filename}\n")
        logf.write(f"# dest: {watch_dir}\n")
        logf.write(f"# expected: {files_total} files, {bytes_total:,} bytes\n")
        logf.write(f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    job = DownloadJob(
        job_id=job_id,
        repo_id=repo_id,
        filename=filename,
        dest_dir=str(watch_dir),
        log_path=str(log_path),
    )
    job.bytes_total = bytes_total
    job.files_total = files_total

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    thread = threading.Thread(
        target=_run_download, args=(job,), daemon=True, name=f"scry-dl-{job_id}"
    )
    thread.start()

    log(f"started download job {job_id}: {repo_id}" + (f" file={filename}" if filename else ""))

    snap = job.snapshot()
    snap["tip"] = (
        f"Poll with download_status(job_id='{job_id}'), "
        f"or run: tail -f {log_path}"
    )
    return snap


def handle_job_status(params: dict) -> dict:
    """Return live status for a download job. Recomputes bytes_done from disk."""
    job_id = params.get("job_id")
    if not job_id:
        # No id given — return all known jobs.
        with _JOBS_LOCK:
            return {"jobs": [j.snapshot() for j in _JOBS.values()]}

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(f"unknown job_id: {job_id}")

    # Refresh bytes_done on demand for in-flight jobs (cheap dir walk).
    if job.state == "downloading":
        job.bytes_done = _walk_dir_size(Path(job.dest_dir))
    return job.snapshot()


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
    "job_status": handle_job_status,
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
