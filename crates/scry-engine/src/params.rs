use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sampler {
    #[default]
    Euler,
    EulerA,
    Heun,
    Dpm2,
    DpmPlusPlus2m,
    DpmPlusPlus2mSde,
    Lcm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateParams {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub model: String,
    #[serde(default)]
    pub loras: Vec<LoraSpec>,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default = "default_cfg")]
    pub cfg_scale: f32,
    #[serde(default)]
    pub sampler: Sampler,
    /// -1 for random.
    #[serde(default = "default_seed")]
    pub seed: i64,
    #[serde(default = "default_batch")]
    pub batch_size: u32,
    #[serde(default)]
    pub vae: Option<String>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraSpec {
    pub model: String,
    #[serde(default = "default_lora_weight")]
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Img2ImgParams {
    #[serde(flatten)]
    pub base: GenerateParams,
    pub input_image: PathBuf,
    /// 0.0 = no change, 1.0 = full denoise.
    #[serde(default = "default_denoise_strength")]
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpscaleParams {
    pub input_image: PathBuf,
    #[serde(default = "default_upscale_factor")]
    pub factor: u32,
    #[serde(default)]
    pub upscaler: Option<String>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub images: Vec<PathBuf>,
    /// The seed actually used (resolved from -1 if random).
    pub seed: i64,
    pub elapsed_ms: u64,
    pub model: String,
}

fn default_width() -> u32 { 1024 }
fn default_height() -> u32 { 1024 }
fn default_steps() -> u32 { 20 }
fn default_cfg() -> f32 { 7.0 }
fn default_seed() -> i64 { -1 }
fn default_batch() -> u32 { 1 }
fn default_lora_weight() -> f32 { 0.8 }
fn default_denoise_strength() -> f32 { 0.7 }
fn default_upscale_factor() -> u32 { 2 }
