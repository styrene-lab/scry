# Scry

Agentic local image generation extension for [Omegon](https://github.com/styrene-lab/omegon) — FLUX, Stable Diffusion, LoRA, and fine-tunes. All inference runs on-device, no external API calls.

## Prerequisites

- **Rust toolchain** — rustc 1.85+ and cargo (via [rustup](https://rustup.rs))
- **Python >= 3.10** — required for the diffusers inference worker
- **At least one diffusion model** checkpoint (FLUX, SDXL, SD1.5)
- **ComfyUI** (optional) — if running locally, scry uses it as a primary backend; otherwise falls back to the built-in diffusers worker

### macOS (nex / nix-darwin)

```sh
nex install rustup python312
# If cargo isn't on PATH after install, ensure ~/.cargo/bin is in your PATH
```

### macOS (Homebrew)

```sh
brew install rustup python@3.12
rustup-init
```

### Linux

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Install python 3.10+ via your distro's package manager
```

## Install

```sh
# As an Omegon extension (builds from source automatically):
omegon extension install https://github.com/styrene-lab/scry.git

# Or manually:
./setup.sh            # build binary + create Python venv + model dirs
./setup.sh --install  # also register with omegon
```

The setup script will:
1. Build the Rust binary (`target/release/scry`)
2. Create a Python venv at `~/.scry/venv` with torch + diffusers
3. Create model directories at `~/.scry/models/`
4. Verify MPS (Apple Silicon) / CUDA support

## Usage

```sh
scry              # Run as Omegon extension (default, v2 protocol)
scry --mcp        # Run as MCP server (Claude Code, Cursor, etc.)
scry --rpc        # Run as Omegon extension (explicit)
```

### With Claude Code

Add to your Claude Code MCP config (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "scry": {
      "command": "/path/to/scry",
      "args": ["--mcp"]
    }
  }
}
```

## What it does

Scry gives Omegon's agent access to local diffusion models for image generation, refinement, and upscaling. It supports two backends — ComfyUI (HTTP, if running) and a built-in diffusers worker (Python sidecar) — and manages model discovery, prompt crafting, and workflow composition.

### Tools

| Tool | Description |
|------|-------------|
| `generate` | Text-to-image generation with optional LoRA stacking |
| `refine` | Image-to-image transformation |
| `upscale` | Super-resolution (2x/4x via RealESRGAN or Lanczos fallback) |
| `list_models` | Discover locally available models |
| `scan_models` | Re-scan model directories |
| `search_models` | Search HuggingFace Hub / CivitAI |
| `download_model` | Download a model from HuggingFace Hub |
| `compose_workflow` | Generate raw ComfyUI API-format workflow JSON |

### Widgets

| Widget | Description |
|--------|-------------|
| Gallery | Generated image history with parameters and outputs |
| Preview | Image preview modal for generation results |
| Models | Local model catalog — checkpoints, LoRAs, VAEs |

### Model discovery

Scry automatically scans these directories at startup:

| Location | Source |
|----------|--------|
| `~/.scry/models/` | Your models (default) |
| `~/.cache/huggingface/hub/` | HuggingFace Hub cache |
| `~/ComfyUI/models/` | ComfyUI (if present) |
| `~/stable-diffusion-webui/models/` | A1111 / Forge (if present) |

Additional directories can be added via `SCRY_EXTRA_MODEL_DIRS` (colon-separated).

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCRY_MODELS_DIR` | `~/.scry/models` | Primary model directory |
| `SCRY_EXTRA_MODEL_DIRS` | — | Additional model dirs (colon-separated) |
| `SCRY_OUTPUT_DIR` | `~/.scry/output` | Where generated images are saved |
| `SCRY_PYTHON` | `~/.scry/venv/bin/python` | Python binary for the diffusers worker |
| `SCRY_WORKER_SCRIPT` | (auto-detected) | Path to `worker.py` |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI API endpoint |

## Development

```sh
cargo build --release
omegon extension install .
```

## License

MIT OR Apache-2.0
