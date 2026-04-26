use std::path::Path;
use std::sync::{Arc, LazyLock};

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use omegon_extension::Result;
use scry_engine::registry::ModelKind;
use scry_engine::{GenerationResult, Pipeline};
use serde_json::{json, Value};

const URI_SCHEME: &str = "scry://artifact/";

/// Run a Pipeline method that produces a `GenerationResult` and wrap the output
/// in an Omegon `ToolResult`-shaped value: inline image content blocks (as
/// `data:` URLs) plus structured metadata under `details`.
macro_rules! run_image_pipeline {
    ($params:expr, $pipeline:expr, $type:ty, $method:ident) => {{
        let parsed: $type = serde_json::from_value($params)
            .map_err(|e| omegon_extension::Error::invalid_params(e.to_string()))?;
        let result = $pipeline
            .$method(parsed)
            .await
            .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))?;
        Ok(image_result_to_tool_result(result))
    }};
}

static TOOLS: LazyLock<Value> = LazyLock::new(|| {
    json!([
        {
            "name": "generate",
            "label": "Generate Image",
            "description": "Generate an image from a text prompt using a local diffusion model. Supports FLUX, Stable Diffusion, SDXL with optional LoRA stacking.",
            "parameters": {
                "type": "object",
                "required": ["prompt", "model"],
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the desired image"
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "What to avoid in the generated image",
                        "default": ""
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name from the registry (use list_models to discover)"
                    },
                    "loras": {
                        "type": "array",
                        "description": "LoRA stack: list of {model, weight} to apply in order",
                        "items": {
                            "type": "object",
                            "required": ["model"],
                            "properties": {
                                "model": { "type": "string", "description": "LoRA name from registry" },
                                "weight": { "type": "number", "default": 0.8, "description": "Blending weight 0.0-1.0" }
                            }
                        },
                        "default": []
                    },
                    "width": { "type": "integer", "default": 1024, "description": "Image width in pixels" },
                    "height": { "type": "integer", "default": 1024, "description": "Image height in pixels" },
                    "steps": { "type": "integer", "default": 20, "description": "Number of denoising steps" },
                    "cfg_scale": { "type": "number", "default": 7.0, "description": "Classifier-free guidance scale" },
                    "sampler": {
                        "type": "string",
                        "enum": ["euler", "euler_a", "heun", "dpm2", "dpm_plus_plus_2m", "dpm_plus_plus_2m_sde", "lcm"],
                        "default": "euler",
                        "description": "Sampling algorithm"
                    },
                    "seed": { "type": "integer", "default": -1, "description": "RNG seed (-1 for random)" },
                    "batch_size": { "type": "integer", "default": 1, "description": "Number of images to generate" },
                    "vae": { "type": "string", "description": "Optional VAE override from registry" }
                }
            }
        },
        {
            "name": "refine",
            "label": "Refine Image",
            "description": "Refine an existing image using img2img. Takes an input image and applies denoising with a text prompt to transform it.",
            "parameters": {
                "type": "object",
                "required": ["prompt", "model", "input_image"],
                "properties": {
                    "prompt": { "type": "string", "description": "Text prompt guiding the refinement" },
                    "negative_prompt": { "type": "string", "default": "" },
                    "model": { "type": "string", "description": "Model name from the registry" },
                    "input_image": { "type": "string", "description": "Path to the input image, or a scry://artifact/<sha256> URI from a previous result." },
                    "strength": {
                        "type": "number", "default": 0.7,
                        "description": "Denoising strength: 0.0 = no change, 1.0 = full denoise"
                    },
                    "loras": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["model"],
                            "properties": {
                                "model": { "type": "string" },
                                "weight": { "type": "number", "default": 0.8 }
                            }
                        },
                        "default": []
                    },
                    "steps": { "type": "integer", "default": 20 },
                    "cfg_scale": { "type": "number", "default": 7.0 },
                    "sampler": { "type": "string", "default": "euler" },
                    "seed": { "type": "integer", "default": -1 }
                }
            }
        },
        {
            "name": "upscale",
            "label": "Upscale Image",
            "description": "Upscale an image using a super-resolution model (e.g. RealESRGAN).",
            "parameters": {
                "type": "object",
                "required": ["input_image"],
                "properties": {
                    "input_image": { "type": "string", "description": "Path to the image to upscale, or a scry://artifact/<sha256> URI from a previous result." },
                    "factor": { "type": "integer", "default": 2, "description": "Upscale factor (2 or 4)" },
                    "upscaler": { "type": "string", "description": "Upscaler model name (default: auto-select)" }
                }
            }
        },
        {
            "name": "list_models",
            "label": "List Models",
            "description": "List all locally available models — checkpoints, LoRAs, VAEs, upscalers. Use this to discover what's available before generating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["checkpoint", "lora", "vae", "upscaler", "text_encoder", "controlnet"],
                        "description": "Filter by model type. Omit to list all."
                    }
                }
            }
        },
        {
            "name": "scan_models",
            "label": "Scan Models",
            "description": "Re-scan the models directory to pick up newly added or removed models.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "compose_workflow",
            "label": "Compose ComfyUI Workflow",
            "description": "Generate a ComfyUI API-format workflow JSON from parameters. Returns valid JSON that can be pasted into ComfyUI or submitted to a running instance. Supports SD/SDXL and FLUX architectures with LoRA stacking.",
            "parameters": {
                "type": "object",
                "required": ["architecture", "checkpoint", "prompt"],
                "properties": {
                    "architecture": {
                        "type": "string",
                        "enum": ["sd", "flux"],
                        "description": "Target architecture: 'sd' for SD1.5/SDXL, 'flux' for FLUX.1/2"
                    },
                    "checkpoint": {
                        "type": "string",
                        "description": "Checkpoint filename (e.g. 'v1-5-pruned.safetensors' or 'flux1-dev.safetensors')"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt for the image"
                    },
                    "negative_prompt": { "type": "string", "default": "", "description": "Negative prompt (SD/SDXL only, ignored for FLUX)" },
                    "loras": {
                        "type": "array",
                        "description": "LoRA stack: [{name, strength_model, strength_clip}]",
                        "items": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "name": { "type": "string", "description": "LoRA filename" },
                                "strength_model": { "type": "number", "default": 1.0 },
                                "strength_clip": { "type": "number", "default": 1.0 }
                            }
                        },
                        "default": []
                    },
                    "width": { "type": "integer", "default": 1024 },
                    "height": { "type": "integer", "default": 1024 },
                    "steps": { "type": "integer", "default": 20 },
                    "cfg": { "type": "number", "default": 7.0, "description": "CFG scale (SD/SDXL) or guidance (FLUX)" },
                    "sampler": { "type": "string", "default": "euler" },
                    "scheduler": { "type": "string", "default": "normal" },
                    "seed": { "type": "integer", "default": -1 },
                    "clip_name1": { "type": "string", "description": "FLUX: T5 encoder filename (default: t5xxl_fp16.safetensors)" },
                    "clip_name2": { "type": "string", "description": "FLUX: CLIP-L filename (default: clip_l.safetensors)" },
                    "vae_name": { "type": "string", "description": "FLUX: VAE filename (default: ae.safetensors)" },
                    "flux_guidance": { "type": "number", "default": 3.5, "description": "FLUX guidance strength" }
                }
            }
        },
        {
            "name": "search_models",
            "label": "Search Models",
            "description": "Search HuggingFace Hub or CivitAI for models. Returns model names, download counts, tags, and base model info.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": { "type": "string", "description": "Search query (e.g. 'flux lora anime', 'sdxl realistic')" },
                    "source": {
                        "type": "string",
                        "enum": ["huggingface", "civitai"],
                        "default": "huggingface",
                        "description": "Which model hub to search"
                    },
                    "type": {
                        "type": "string",
                        "description": "Filter by model type. CivitAI: Checkpoint, LORA, ControlNet. HF: text-to-image."
                    },
                    "limit": { "type": "integer", "default": 20, "description": "Max results to return" }
                }
            }
        },
        {
            "name": "download_model",
            "label": "Download Model",
            "description": "Start a HuggingFace download in the background. Returns immediately with a job_id and log_path; poll download_status (or tail the log file) to track progress.",
            "parameters": {
                "type": "object",
                "required": ["repo_id"],
                "properties": {
                    "repo_id": { "type": "string", "description": "HuggingFace repo ID (e.g. 'black-forest-labs/FLUX.1-schnell')" },
                    "filename": { "type": "string", "description": "Specific file to download (e.g. 'flux1-schnell.safetensors'). Omit for full model." },
                    "dest_dir": { "type": "string", "description": "Destination directory. Defaults to HF cache." }
                }
            }
        },
        {
            "name": "download_status",
            "label": "Download Status",
            "description": "Poll a download job started by download_model. Returns state (pending|downloading|complete|failed), bytes_done, bytes_total, log_path, and result_path on completion. Omit job_id to list all known jobs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": { "type": "string", "description": "Job ID returned from download_model. Omit to list all jobs." }
                }
            }
        },
        {
            "name": "get_artifact",
            "label": "Get Artifact",
            "description": "Fetch a previously-generated image by its content-address (sha256). Accepts either a bare sha256 or a scry://artifact/<sha256> URI. Returns the image as an inline content block plus its sidecar metadata. Use this when a remote client needs to retrieve an artifact that's no longer in the conversation.",
            "parameters": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "sha256 hex (or scry://artifact/<sha256> URI) of the artifact"
                    }
                }
            }
        }
    ])
});

pub fn tool_definitions() -> Value {
    TOOLS.clone()
}

pub async fn dispatch_tool(
    tool_name: &str,
    params: Value,
    pipeline: &Arc<Pipeline>,
) -> Result<Value> {
    match tool_name {
        "generate" => run_image_pipeline!(params, pipeline, scry_engine::GenerateParams, generate),
        "refine" => run_image_pipeline!(params, pipeline, scry_engine::Img2ImgParams, img2img),
        "upscale" => run_image_pipeline!(params, pipeline, scry_engine::UpscaleParams, upscale),
        "list_models" => handle_list_models(params, pipeline).await,
        "scan_models" => handle_scan_models(pipeline).await,
        "compose_workflow" => handle_compose_workflow(params),
        "search_models" => handle_search_models(params, pipeline).await,
        "download_model" => handle_download_model(params, pipeline).await,
        "download_status" => handle_download_status(params, pipeline).await,
        "get_artifact" => handle_get_artifact(params, pipeline),
        _ => Err(omegon_extension::Error::method_not_found(tool_name)),
    }
}

fn handle_get_artifact(params: Value, pipeline: &Arc<Pipeline>) -> Result<Value> {
    let raw = params
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| omegon_extension::Error::invalid_params("missing 'id'"))?;
    let id = raw.strip_prefix(URI_SCHEME).unwrap_or(raw);

    let (path, meta) = pipeline
        .get_artifact(id)
        .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))?;

    let url = read_image_as_data_url(&path, &meta.media_type)
        .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))?;

    Ok(json!({
        "content": [
            {
                "type": "image",
                "url": url,
                "media_type": meta.media_type,
            },
            {
                "type": "text",
                "text": format!(
                    "id={} model={} seed={} bytes={} created_at={}",
                    meta.id,
                    meta.model.as_deref().unwrap_or("?"),
                    meta.seed.map(|s| s.to_string()).unwrap_or_else(|| "?".into()),
                    meta.bytes,
                    meta.created_at,
                ),
            }
        ],
        "details": {
            "id": meta.id,
            "uri": format!("{URI_SCHEME}{}", meta.id),
            "path": path,
            "media_type": meta.media_type,
            "bytes": meta.bytes,
            "model": meta.model,
            "seed": meta.seed,
            "prompt": meta.prompt,
            "created_at": meta.created_at,
            "source_tool": meta.source_tool,
            "params": meta.params,
        }
    }))
}

async fn handle_list_models(params: Value, pipeline: &Arc<Pipeline>) -> Result<Value> {
    let kind: Option<ModelKind> = params
        .get("kind")
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    let models = pipeline.list_models(kind).await;

    serde_json::to_value(models)
        .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))
}

fn handle_compose_workflow(params: Value) -> Result<Value> {
    let arch = params
        .get("architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("sd");

    let wp: scry_engine::WorkflowParams = serde_json::from_value(params.clone())
        .map_err(|e| omegon_extension::Error::invalid_params(e.to_string()))?;

    let workflow = match arch {
        "flux" => scry_engine::flux_txt2img(&wp),
        _ => scry_engine::sd_txt2img(&wp),
    };

    let workflow_json = serde_json::to_value(&workflow)
        .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))?;

    Ok(json!({
        "architecture": arch,
        "workflow": workflow_json,
        "prompt_body": workflow.to_prompt_body(None),
        "node_count": workflow.nodes.len(),
    }))
}

async fn handle_search_models(params: Value, pipeline: &Arc<Pipeline>) -> Result<Value> {
    let source = params
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("huggingface");

    let method = match source {
        "civitai" => "civitai_search",
        _ => "hf_search",
    };

    pipeline
        .worker_rpc(method, params)
        .await
        .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))
}

async fn handle_download_model(params: Value, pipeline: &Arc<Pipeline>) -> Result<Value> {
    pipeline
        .worker_rpc("hf_download", params)
        .await
        .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))
}

async fn handle_download_status(params: Value, pipeline: &Arc<Pipeline>) -> Result<Value> {
    pipeline
        .worker_rpc("job_status", params)
        .await
        .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))
}

async fn handle_scan_models(pipeline: &Arc<Pipeline>) -> Result<Value> {
    let count = pipeline
        .scan_models()
        .await
        .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))?;

    Ok(json!({
        "scanned": true,
        "models_found": count
    }))
}

/// Convert a `GenerationResult` into an Omegon `ToolResult`-shaped value.
///
/// `content` carries one image block per output, with the image bytes inlined as a
/// `data:` URL — matching omegon's convention so the host TUI (`ratatui-image`) and the
/// MCP shim (which forwards `url` into MCP's image `data` field) can both render it
/// without filesystem access. `details.artifacts[]` carries each output's content-address
/// (sha256) and `scry://artifact/<sha>` URI, which refine/upscale and `get_artifact`
/// accept in place of a path.
fn image_result_to_tool_result(result: GenerationResult) -> Value {
    let mut content = Vec::with_capacity(result.images.len() + 1);
    for path in &result.images {
        let media_type = media_type_for(path);
        match read_image_as_data_url(path, media_type) {
            Ok(url) => content.push(json!({
                "type": "image",
                "url": url,
                "media_type": media_type,
            })),
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "could not inline image");
                content.push(json!({
                    "type": "text",
                    "text": format!("[image at {} could not be inlined: {e}]", path.display()),
                }));
            }
        }
    }
    let artifact_summary: Vec<String> = result
        .artifacts
        .iter()
        .map(|a| a.uri.clone())
        .collect();
    let path_summary: Vec<String> = result
        .images
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    // The MCP shim forwards only `content` to MCP-only clients (Claude Code etc.) and drops
    // `details`, so the metadata gets a text block too — otherwise MCP callers lose the seed,
    // artifact URIs, and timing info that previous callers relied on.
    let summary_id_field = if artifact_summary.is_empty() {
        format!("paths={}", path_summary.join(", "))
    } else {
        format!("artifacts={}", artifact_summary.join(", "))
    };
    content.push(json!({
        "type": "text",
        "text": format!(
            "model={} seed={} elapsed_ms={} {}",
            result.model,
            result.seed,
            result.elapsed_ms,
            summary_id_field,
        ),
    }));
    json!({
        "content": content,
        "details": {
            "images": result.images,
            "artifacts": result.artifacts,
            "seed": result.seed,
            "elapsed_ms": result.elapsed_ms,
            "model": result.model,
        }
    })
}

fn read_image_as_data_url(path: &Path, media_type: &str) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(format!("data:{media_type};base64,{}", B64.encode(&bytes)))
}

fn media_type_for(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        // Diffusers/MLX paths in scry always emit PNG today; this is the safe default.
        _ => "image/png",
    }
}
