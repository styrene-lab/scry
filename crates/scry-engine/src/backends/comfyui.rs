use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::backend::{Architecture, Backend, BackendStatus};
use crate::params::{GenerateParams, GenerationResult, Img2ImgParams, UpscaleParams};
use crate::workflow_templates::{self, LoraParam, WorkflowParams};
use crate::{Error, Result};

/// Backend that submits ComfyUI API-format workflows to a running ComfyUI instance.
///
/// Completely optional — if ComfyUI isn't running, health_check returns unavailable
/// and the pipeline falls through to other backends.
pub struct ComfyUiBackend {
    base_url: String,
    client: reqwest::Client,
    #[allow(dead_code)] // Will be used for downloading generated images.
    output_dir: PathBuf,
}

impl ComfyUiBackend {
    pub fn new(base_url: String, output_dir: PathBuf) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(2))
            .build()
            .expect("failed to build HTTP client");

        Self {
            base_url,
            client,
            output_dir,
        }
    }

    /// Submit a workflow and poll until complete. Returns output image filenames.
    async fn execute_workflow(&self, workflow: &crate::workflow::Workflow) -> Result<Vec<String>> {
        let body = workflow.to_prompt_body(None);

        let resp = self
            .client
            .post(format!("{}/prompt", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::BackendUnavailable(format!("ComfyUI POST /prompt failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::GenerationFailed(format!(
                "ComfyUI returned {status}: {text}"
            )));
        }

        let prompt_resp: Value = resp
            .json()
            .await
            .map_err(|e| Error::GenerationFailed(format!("invalid ComfyUI response: {e}")))?;

        let prompt_id = prompt_resp
            .get("prompt_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::GenerationFailed("missing prompt_id in response".to_string()))?
            .to_string();

        debug!(prompt_id = %prompt_id, "workflow submitted to ComfyUI");

        // Poll /history/{prompt_id} until results appear.
        let mut images = Vec::new();
        let poll_interval = Duration::from_millis(500);
        let max_polls = 600; // 5 minutes max

        for _ in 0..max_polls {
            tokio::time::sleep(poll_interval).await;

            let history = self
                .client
                .get(format!("{}/history/{}", self.base_url, prompt_id))
                .send()
                .await
                .map_err(|e| {
                    Error::GenerationFailed(format!("ComfyUI history poll failed: {e}"))
                })?;

            if !history.status().is_success() {
                continue;
            }

            let history_json: Value = history.json().await.unwrap_or_default();

            if let Some(entry) = history_json.get(&prompt_id) {
                if let Some(outputs) = entry.get("outputs") {
                    // Find all SaveImage / PreviewImage node outputs.
                    if let Some(outputs_map) = outputs.as_object() {
                        for (_node_id, node_output) in outputs_map {
                            if let Some(img_list) = node_output.get("images").and_then(|v| v.as_array()) {
                                for img in img_list {
                                    let filename = img.get("filename").and_then(|v| v.as_str()).unwrap_or("");
                                    let subfolder = img.get("subfolder").and_then(|v| v.as_str()).unwrap_or("");
                                    if !filename.is_empty() {
                                        let path = if subfolder.is_empty() {
                                            filename.to_string()
                                        } else {
                                            format!("{subfolder}/{filename}")
                                        };
                                        images.push(path);
                                    }
                                }
                            }
                        }
                    }

                    if !images.is_empty() {
                        debug!(prompt_id = %prompt_id, count = images.len(), "ComfyUI generation complete");
                        return Ok(images);
                    }
                }

                // Check for errors in execution status.
                if let Some(status) = entry.get("status") {
                    if let Some(msgs) = status.get("messages").and_then(|v| v.as_array()) {
                        for msg in msgs {
                            if let Some(arr) = msg.as_array() {
                                if arr.first().and_then(|v| v.as_str()) == Some("execution_error") {
                                    let detail = arr.get(1).map(|v| v.to_string()).unwrap_or_default();
                                    return Err(Error::GenerationFailed(format!(
                                        "ComfyUI execution error: {detail}"
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }

        Err(Error::GenerationFailed(
            "ComfyUI generation timed out after 5 minutes".to_string(),
        ))
    }

    fn to_workflow_params(&self, params: &GenerateParams) -> WorkflowParams {
        WorkflowParams {
            prompt: params.prompt.clone(),
            negative_prompt: params.negative_prompt.clone(),
            checkpoint: params.model.clone(),
            loras: params
                .loras
                .iter()
                .map(|l| LoraParam {
                    name: l.model.clone(),
                    strength_model: l.weight as f64,
                    strength_clip: l.weight as f64,
                })
                .collect(),
            width: params.width,
            height: params.height,
            steps: params.steps,
            cfg: params.cfg_scale as f64,
            sampler: "euler".to_string(),
            scheduler: "normal".to_string(),
            seed: params.seed,
            denoise: 1.0,
            filename_prefix: "scry_comfyui".to_string(),
            clip_name1: String::new(),
            clip_name2: String::new(),
            vae_name: String::new(),
            flux_guidance: 3.5,
        }
    }

    /// Guess whether to use FLUX or SD template based on model name.
    fn is_flux(model: &str) -> bool {
        let lower = model.to_lowercase();
        lower.contains("flux")
    }
}

#[async_trait]
impl Backend for ComfyUiBackend {
    fn name(&self) -> &str {
        "comfyui"
    }

    fn supported_architectures(&self) -> &[Architecture] {
        // ComfyUI supports everything via its node system.
        &[
            Architecture::Sd15,
            Architecture::Sd21,
            Architecture::Sdxl,
            Architecture::Sd3,
            Architecture::Flux,
            Architecture::Flux2,
        ]
    }

    async fn health_check(&self) -> Result<BackendStatus> {
        match self
            .client
            .get(format!("{}/system_stats", self.base_url))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                let stats: Value = resp.json().await.unwrap_or_default();
                let device = stats
                    .pointer("/system/devices/0/name")
                    .and_then(|v| v.as_str())
                    .map(String::from);

                info!(url = %self.base_url, "ComfyUI is available");
                Ok(BackendStatus {
                    available: true,
                    name: "comfyui".to_string(),
                    version: None,
                    device,
                })
            }
            Ok(resp) => {
                warn!(url = %self.base_url, status = %resp.status(), "ComfyUI returned error");
                Ok(BackendStatus {
                    available: false,
                    name: "comfyui".to_string(),
                    version: None,
                    device: None,
                })
            }
            Err(_) => Ok(BackendStatus {
                available: false,
                name: "comfyui".to_string(),
                version: None,
                device: None,
            }),
        }
    }

    async fn generate(&self, params: &GenerateParams) -> Result<GenerationResult> {
        let wp = self.to_workflow_params(params);
        let workflow = if Self::is_flux(&params.model) {
            workflow_templates::flux_txt2img(&wp)
        } else {
            workflow_templates::sd_txt2img(&wp)
        };

        let t0 = std::time::Instant::now();
        let image_names = self.execute_workflow(&workflow).await?;
        let elapsed_ms = t0.elapsed().as_millis() as u64;

        // ComfyUI saves images to its own output dir.
        // Return filenames — the user/agent can fetch via GET /view.
        let images: Vec<PathBuf> = image_names.iter().map(PathBuf::from).collect();

        Ok(GenerationResult {
            images,
            seed: params.seed,
            elapsed_ms,
            model: params.model.clone(),
        })
    }

    async fn img2img(&self, _params: &Img2ImgParams) -> Result<GenerationResult> {
        Err(Error::GenerationFailed(
            "ComfyUI img2img not yet implemented — use diffusers backend".to_string(),
        ))
    }

    async fn upscale(&self, _params: &UpscaleParams) -> Result<GenerationResult> {
        Err(Error::GenerationFailed(
            "ComfyUI upscale not yet implemented — use diffusers backend".to_string(),
        ))
    }
}
