//! Pre-built workflow templates for common generation patterns.
//!
//! Each template produces a valid ComfyUI API-format [`Workflow`] that can be
//! serialized to JSON and submitted to POST /prompt, or returned to the agent
//! for inspection/modification.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::workflow::{link, Workflow, WorkflowBuilder};

/// Parameters shared across all workflow templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowParams {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub checkpoint: String,
    #[serde(default)]
    pub loras: Vec<LoraParam>,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default = "default_cfg")]
    pub cfg: f64,
    #[serde(default = "default_sampler")]
    pub sampler: String,
    #[serde(default = "default_scheduler")]
    pub scheduler: String,
    #[serde(default = "default_seed")]
    pub seed: i64,
    #[serde(default = "default_denoise")]
    pub denoise: f64,
    #[serde(default)]
    pub filename_prefix: String,
    // FLUX-specific
    #[serde(default)]
    pub clip_name1: String,
    #[serde(default)]
    pub clip_name2: String,
    #[serde(default)]
    pub vae_name: String,
    #[serde(default = "default_flux_guidance")]
    pub flux_guidance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraParam {
    pub name: String,
    #[serde(default = "default_lora_strength")]
    pub strength_model: f64,
    #[serde(default = "default_lora_strength")]
    pub strength_clip: f64,
}

fn default_width() -> u32 { 1024 }
fn default_height() -> u32 { 1024 }
fn default_steps() -> u32 { 20 }
fn default_cfg() -> f64 { 7.0 }
fn default_sampler() -> String { "euler".to_string() }
fn default_scheduler() -> String { "normal".to_string() }
fn default_seed() -> i64 { -1 }
fn default_denoise() -> f64 { 1.0 }
fn default_flux_guidance() -> f64 { 3.5 }
fn default_lora_strength() -> f64 { 1.0 }

/// SD 1.5 / SDXL txt2img workflow.
///
/// Topology: CheckpointLoaderSimple → [LoRA chain] → CLIPTextEncode(×2) → KSampler → VAEDecode → SaveImage
pub fn sd_txt2img(params: &WorkflowParams) -> Workflow {
    let mut b = WorkflowBuilder::new();

    let ckpt = b.node("CheckpointLoaderSimple", &[
        ("ckpt_name", json!(params.checkpoint)),
    ]);

    // LoRA chain — each takes MODEL+CLIP, outputs modified MODEL+CLIP.
    let (model_src, clip_src) = apply_lora_chain(&mut b, &ckpt, 0, &ckpt, 1, &params.loras);

    let positive = b.node("CLIPTextEncode", &[
        ("text", json!(params.prompt)),
        ("clip", link(&clip_src, 1)),
    ]);

    let negative = b.node("CLIPTextEncode", &[
        ("text", json!(params.negative_prompt)),
        ("clip", link(&clip_src, 1)),
    ]);

    let latent = b.node("EmptyLatentImage", &[
        ("width", json!(params.width)),
        ("height", json!(params.height)),
        ("batch_size", json!(1)),
    ]);

    let seed = resolve_seed(params.seed);

    let sampled = b.node("KSampler", &[
        ("model", link(&model_src, 0)),
        ("positive", link(&positive, 0)),
        ("negative", link(&negative, 0)),
        ("latent_image", link(&latent, 0)),
        ("seed", json!(seed)),
        ("steps", json!(params.steps)),
        ("cfg", json!(params.cfg)),
        ("sampler_name", json!(params.sampler)),
        ("scheduler", json!(params.scheduler)),
        ("denoise", json!(params.denoise)),
    ]);

    let decoded = b.node("VAEDecode", &[
        ("samples", link(&sampled, 0)),
        ("vae", link(&ckpt, 2)),
    ]);

    let prefix = if params.filename_prefix.is_empty() {
        "scry"
    } else {
        &params.filename_prefix
    };

    b.node("SaveImage", &[
        ("images", link(&decoded, 0)),
        ("filename_prefix", json!(prefix)),
    ]);

    b.build()
}

/// FLUX txt2img workflow.
///
/// Topology: UNETLoader + DualCLIPLoader + VAELoader → CLIPTextEncode → FluxGuidance →
///           RandomNoise + BasicGuider + KSamplerSelect + BasicScheduler →
///           SamplerCustomAdvanced → VAEDecode → SaveImage
pub fn flux_txt2img(params: &WorkflowParams) -> Workflow {
    let mut b = WorkflowBuilder::new();

    let unet = b.node("UNETLoader", &[
        ("unet_name", json!(params.checkpoint)),
        ("weight_dtype", json!("default")),
    ]);

    let clip1 = if params.clip_name1.is_empty() { "t5xxl_fp16.safetensors" } else { &params.clip_name1 };
    let clip2 = if params.clip_name2.is_empty() { "clip_l.safetensors" } else { &params.clip_name2 };

    let clip = b.node("DualCLIPLoader", &[
        ("clip_name1", json!(clip1)),
        ("clip_name2", json!(clip2)),
        ("type", json!("flux")),
    ]);

    let vae_name = if params.vae_name.is_empty() { "ae.safetensors" } else { &params.vae_name };
    let vae = b.node("VAELoader", &[
        ("vae_name", json!(vae_name)),
    ]);

    // LoRA chain on the unet model. FLUX LoRAs take MODEL only (no CLIP adaptation typically),
    // but LoraLoader still expects both — pass clip through.
    let (model_src, _clip_src) = apply_lora_chain(&mut b, &unet, 0, &clip, 0, &params.loras);

    let encoded = b.node("CLIPTextEncode", &[
        ("text", json!(params.prompt)),
        ("clip", link(&clip, 0)),
    ]);

    let guided = b.node("FluxGuidance", &[
        ("conditioning", link(&encoded, 0)),
        ("guidance", json!(params.flux_guidance)),
    ]);

    let seed = resolve_seed(params.seed);

    let noise = b.node("RandomNoise", &[
        ("noise_seed", json!(seed)),
    ]);

    let guider = b.node("BasicGuider", &[
        ("model", link(&model_src, 0)),
        ("conditioning", link(&guided, 0)),
    ]);

    let sampler = b.node("KSamplerSelect", &[
        ("sampler_name", json!(params.sampler)),
    ]);

    let sigmas = b.node("BasicScheduler", &[
        ("model", link(&model_src, 0)),
        ("scheduler", json!(params.scheduler)),
        ("steps", json!(params.steps)),
        ("denoise", json!(params.denoise)),
    ]);

    let latent = b.node("EmptySD3LatentImage", &[
        ("width", json!(params.width)),
        ("height", json!(params.height)),
        ("batch_size", json!(1)),
    ]);

    let sampled = b.node("SamplerCustomAdvanced", &[
        ("noise", link(&noise, 0)),
        ("guider", link(&guider, 0)),
        ("sampler", link(&sampler, 0)),
        ("sigmas", link(&sigmas, 0)),
        ("latent_image", link(&latent, 0)),
    ]);

    let decoded = b.node("VAEDecode", &[
        ("samples", link(&sampled, 0)),
        ("vae", link(&vae, 0)),
    ]);

    let prefix = if params.filename_prefix.is_empty() {
        "scry_flux"
    } else {
        &params.filename_prefix
    };

    b.node("SaveImage", &[
        ("images", link(&decoded, 0)),
        ("filename_prefix", json!(prefix)),
    ]);

    b.build()
}

/// Insert a LoRA chain between a model source and clip source.
/// Returns (final_model_node_id, final_clip_node_id) — use output index 0 for model, 1 for clip.
fn apply_lora_chain(
    b: &mut WorkflowBuilder,
    model_node: &str,
    model_output: u32,
    clip_node: &str,
    clip_output: u32,
    loras: &[LoraParam],
) -> (String, String) {
    if loras.is_empty() {
        return (model_node.to_string(), clip_node.to_string());
    }

    let mut current_model = model_node.to_string();
    let mut current_model_out = model_output;
    let mut current_clip = clip_node.to_string();
    let mut current_clip_out = clip_output;

    for lora in loras {
        let id = b.node("LoraLoader", &[
            ("model", link(&current_model, current_model_out)),
            ("clip", link(&current_clip, current_clip_out)),
            ("lora_name", json!(lora.name)),
            ("strength_model", json!(lora.strength_model)),
            ("strength_clip", json!(lora.strength_clip)),
        ]);
        current_model = id.clone();
        current_model_out = 0;
        current_clip = id;
        current_clip_out = 1;
    }

    (current_model, current_clip)
}

fn resolve_seed(seed: i64) -> u64 {
    if seed < 0 {
        // Random seed — use a deterministic placeholder.
        // ComfyUI will randomize if the node supports it, but for API format
        // we need a concrete value. Use current time as entropy.
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64 % (1u64 << 53))
            .unwrap_or(42)
    } else {
        seed as u64
    }
}
