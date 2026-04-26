# Scry — Scope and Boundary

This document fixes the boundary between **Scry** and **Omegon's `render` extension**. It exists because the two projects overlap on "produce a visual artifact for the user" and we don't want to duplicate work or surprise contributors with parallel code paths. Read this before adding any new generation tool.

## One-line distinction

> **`render` is for deterministic, structured visuals (D2, Excalidraw, native SVG diagrams, opinionated MLX-FLUX presets) that should run anywhere Omegon runs. Scry is for heavy, customizable diffusion (full model registry, LoRA stacks, samplers, fine-tunes) that wants to live where the GPU lives and serve clients remotely.**

## What lives where

| Concern | Home | Notes |
| --- | --- | --- |
| D2 diagrams | `render` | `render_diagram` tool; CLI-driven. |
| Excalidraw | `render` | Playwright + headless Chromium; only place we tolerate a browser dep. |
| Native SVG architecture / flow / motif diagrams | `render` | Motif-based specs → SVG → PNG via `rsvg-convert`. |
| Opinionated FLUX presets (`schnell`, `dev`, `diagram`, `portrait`, `wide`) | `render` | MLX, Apple-Silicon-only; small parameter surface. |
| Arbitrary diffusion model selection (SD1.5/2.1/SDXL/SD3/FLUX/FLUX2) | `scry` | Through the registry. |
| LoRA stacking (multiple LoRAs, per-LoRA weight) | `scry` | First-class. |
| Custom samplers, schedulers, CFG, step counts | `scry` | Full diffusers/ComfyUI parameter surface. |
| Img2img refine, super-resolution upscaling | `scry` | `refine`, `upscale` tools. |
| HuggingFace / CivitAI search, downloads, registry scanning | `scry` | `search_models`, `download_model`, `scan_models`. |
| Fine-tune / training workflows | `scry` (future) | If/when we add training, it belongs here. |
| Remote-MCP serving, content-addressed artifact store | `scry` (future) | Heavy compute is the case for remote; render stays local. |
| Logos, UI mockups, reference architectures | **decide per-output** | Vector/structured → `render`. Photoreal/illustrative styling → `scry`, optionally consuming a `render` SVG as init image. |

## Decision rules for new tools

When proposing a new visual tool, walk this list:

1. **Is the output deterministic given the input?** (Same spec → same pixels.) If yes, it belongs in `render`. Diffusion is non-deterministic by nature; use `scry` only when stochasticity is desirable.
2. **Does it need a GPU / >5 GB of model weights / >10 s of compute?** If yes, `scry`. If it runs in <1 s on a laptop, `render`.
3. **Is the natural primary output a vector format (SVG)?** Then `render`. Scry produces raster (PNG).
4. **Does the user iterate by editing structured spec text vs. by re-rolling seeds and tweaking prompts?** Spec-text iteration → `render`. Prompt/seed iteration → `scry`.
5. **Does it require a model from HuggingFace/CivitAI?** Then `scry` (because of registry, downloads, LoRA support).

If two of those answers point opposite directions, the tool is probably a **composition** of both extensions — e.g. "logo idea generation" might fan out: `scry` produces stylistic concepts; `render` produces the cleaned-up SVG. Implement each piece in its proper home and orchestrate at the agent layer, not by reimplementing one in the other.

## Shared contracts

Both extensions return Omegon's `ToolResult` shape:

```jsonc
{
  "content": [
    { "type": "image", "url": "data:image/png;base64,…", "media_type": "image/png" },
    { "type": "text", "text": "human-readable summary" }
  ],
  "details": { /* structured metadata: paths, seeds, timings, parameters */ }
}
```

This is the only image-return shape we ship. The MCP shim in `omegon-extension` forwards `content` to standard MCP clients and the host TUI renders the data URL via `ratatui-image`. Do not invent a new shape — extend `details` instead.

## What this implies for contributors

- **Don't add a `generate_diagram` tool to scry.** Use `render`.
- **Don't add LoRA support to render.** Use `scry`.
- **Don't add a Playwright/Chromium dep to scry** unless a future use case can't be served by `render` *and* the boundary above gets revised in this doc first.
- **Keep the Alpharius palette consistent** when scry produces outputs intended to compose with `render`'s deterministic visuals (`#3b82f6` primary, `#c2410c` start, `#047857` end, `#b45309` decision, `#6d28d9` AI).
- **The remote-MCP and artifact-store work is a scry concern.** `render` stays a local-first extension.

## Open questions (revisit when relevant)

- **Logo generation.** Currently no clean owner. Likely composition of `scry` (concept) + `render` (vectorize). When we tackle it, write the workflow down here.
- **Image-to-spec extraction** (e.g., "regenerate this hand-drawn diagram as a clean D2 spec"). Probably a third path; revisit when there's a real use case.
- **Animation / video.** Out of scope for both today; if it lands, it's almost certainly `scry`'s problem.
