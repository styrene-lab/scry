use std::sync::{Arc, LazyLock};

use omegon_extension::Result;
use scry_engine::registry::ModelKind;
use scry_engine::Pipeline;
use serde_json::{json, Value};

macro_rules! run_pipeline {
    ($params:expr, $pipeline:expr, $type:ty, $method:ident) => {{
        let parsed: $type = serde_json::from_value($params)
            .map_err(|e| omegon_extension::Error::invalid_params(e.to_string()))?;
        let result = $pipeline
            .$method(parsed)
            .await
            .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))?;
        serde_json::to_value(result)
            .map_err(|e| omegon_extension::Error::internal_error(e.to_string()))
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
                    "input_image": { "type": "string", "description": "Path to the input image" },
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
                    "input_image": { "type": "string", "description": "Path to the image to upscale" },
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
            "description": "Download a model from HuggingFace Hub. Specify repo_id and optionally a specific filename.",
            "parameters": {
                "type": "object",
                "required": ["repo_id"],
                "properties": {
                    "repo_id": { "type": "string", "description": "HuggingFace repo ID (e.g. 'black-forest-labs/FLUX.1-schnell')" },
                    "filename": { "type": "string", "description": "Specific file to download (e.g. 'flux1-schnell.safetensors'). Omit for full model." },
                    "dest_dir": { "type": "string", "description": "Destination directory. Defaults to HF cache." }
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
        "generate" => run_pipeline!(params, pipeline, scry_engine::GenerateParams, generate),
        "refine" => run_pipeline!(params, pipeline, scry_engine::Img2ImgParams, img2img),
        "upscale" => run_pipeline!(params, pipeline, scry_engine::UpscaleParams, upscale),
        "list_models" => handle_list_models(params, pipeline).await,
        "scan_models" => handle_scan_models(pipeline).await,
        "compose_workflow" => handle_compose_workflow(params),
        "search_models" => handle_search_models(params, pipeline).await,
        "download_model" => handle_download_model(params, pipeline).await,
        _ => Err(omegon_extension::Error::method_not_found(tool_name)),
    }
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
