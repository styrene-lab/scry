# Scry

Agentic local image generation extension for [Omegon](https://github.com/styrene-lab/omegon) — FLUX, Stable Diffusion, LoRA, and fine-tunes. All inference runs on-device, no external API calls.

## Install

```sh
omegon extension install https://github.com/styrene-lab/scry.git
```

Requires a Rust toolchain — the extension builds from source on install.

## What it does

Scry gives Omegon's agent access to local diffusion models for image generation, refinement, and upscaling. It talks to ComfyUI under the hood and manages model discovery, prompt crafting, and workflow composition.

### Tools

| Tool | Description |
|------|-------------|
| `generate` | Text-to-image generation |
| `refine` | Image-to-image transformation |
| `upscale` | Super-resolution (2x/4x) |
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

## Prerequisites

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally
- At least one diffusion model checkpoint (FLUX, SDXL, SD1.5)

## Development

```sh
cargo build --release
omegon extension install .
```

## License

MIT OR Apache-2.0
