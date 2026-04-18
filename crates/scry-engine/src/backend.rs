use async_trait::async_trait;
use serde_json::Value;

use crate::params::{GenerateParams, GenerationResult, Img2ImgParams, UpscaleParams};
use crate::{Error, Result};

#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn supported_architectures(&self) -> &[Architecture];
    async fn health_check(&self) -> Result<BackendStatus>;
    async fn generate(&self, params: &GenerateParams) -> Result<GenerationResult>;
    async fn img2img(&self, params: &Img2ImgParams) -> Result<GenerationResult>;
    async fn upscale(&self, params: &UpscaleParams) -> Result<GenerationResult>;

    /// Pass-through RPC for non-generation operations (search, download, etc.).
    /// Only backends with a worker process implement this.
    async fn worker_rpc(&self, _method: &str, _params: Value) -> Result<Value> {
        Err(Error::BackendUnavailable(
            format!("{} does not support worker_rpc", self.name()),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    Sd15,
    Sd21,
    Sdxl,
    Sd3,
    Flux,
    Flux2,
}

impl Architecture {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Sd15 => "sd1.5",
            Self::Sd21 => "sd2.1",
            Self::Sdxl => "sdxl",
            Self::Sd3 => "sd3",
            Self::Flux => "flux",
            Self::Flux2 => "flux2",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendStatus {
    pub available: bool,
    pub name: String,
    pub version: Option<String>,
    pub device: Option<String>,
}
