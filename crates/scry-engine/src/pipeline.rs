use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::backend::Backend;
use crate::params::{GenerateParams, GenerationResult, Img2ImgParams, UpscaleParams};
use crate::registry::{ModelKind, ModelRegistry};
use crate::store::{ArtifactMeta, ArtifactStore, MetaBuilder};
use crate::{Error, Result};

pub struct Pipeline {
    registry: RwLock<ModelRegistry>,
    backends: Vec<BackendSlot>,
    output_dir: PathBuf,
    store: ArtifactStore,
}

struct BackendSlot {
    backend: Box<dyn Backend>,
    healthy: AtomicBool,
}

impl Pipeline {
    /// Create a new pipeline.
    ///
    /// `extra_model_dirs` — additional directories to scan beyond the auto-discovered
    /// HuggingFace cache, ComfyUI, and A1111 locations.
    /// `artifacts_dir` — root for the content-addressed artifact store.
    pub fn new(extra_model_dirs: &[PathBuf], output_dir: PathBuf, artifacts_dir: PathBuf) -> Self {
        Self {
            registry: RwLock::new(ModelRegistry::new(extra_model_dirs)),
            backends: Vec::new(),
            output_dir,
            store: ArtifactStore::new(artifacts_dir),
        }
    }

    pub fn store(&self) -> &ArtifactStore {
        &self.store
    }

    /// Look up an artifact by sha256 hex.
    pub fn get_artifact(&self, id: &str) -> Result<(PathBuf, ArtifactMeta)> {
        self.store.get(id)
    }

    pub fn add_backend(&mut self, backend: Box<dyn Backend>) {
        info!(name = backend.name(), "registered backend");
        self.backends.push(BackendSlot {
            backend,
            healthy: AtomicBool::new(false),
        });
    }

    pub async fn scan_models(&self) -> Result<usize> {
        let mut reg = self.registry.write().await;
        reg.scan()
    }

    pub async fn list_models(&self, kind: Option<ModelKind>) -> Vec<crate::registry::ModelEntry> {
        let reg = self.registry.read().await;
        reg.list_owned(kind)
    }

    pub async fn generate(&self, mut params: GenerateParams) -> Result<GenerationResult> {
        if params.output_dir.is_none() {
            params.output_dir = Some(self.output_dir.clone());
        }

        let (resolved_model, arch) = self.resolve_model(&params.model).await;
        params.model = resolved_model;

        let backend = self.select_backend(arch).await?;
        let mut result = backend.generate(&params).await?;
        let params_json = serde_json::to_value(&params).unwrap_or(serde_json::Value::Null);
        self.attach_artifacts(
            &mut result,
            "generate",
            Some(params.prompt.clone()),
            params_json,
        );
        Ok(result)
    }

    pub async fn img2img(&self, mut params: Img2ImgParams) -> Result<GenerationResult> {
        if params.base.output_dir.is_none() {
            params.base.output_dir = Some(self.output_dir.clone());
        }

        // Accept `scry://artifact/<sha>` for input_image, in addition to plain paths.
        let resolved_input = self
            .store
            .resolve(&params.input_image.to_string_lossy())?;
        params.input_image = resolved_input;

        let (resolved_model, arch) = self.resolve_model(&params.base.model).await;
        params.base.model = resolved_model;

        let backend = self.select_backend(arch).await?;
        let mut result = backend.img2img(&params).await?;
        let params_json = serde_json::to_value(&params).unwrap_or(serde_json::Value::Null);
        self.attach_artifacts(
            &mut result,
            "refine",
            Some(params.base.prompt.clone()),
            params_json,
        );
        Ok(result)
    }

    pub async fn upscale(&self, mut params: UpscaleParams) -> Result<GenerationResult> {
        if params.output_dir.is_none() {
            params.output_dir = Some(self.output_dir.clone());
        }

        let resolved_input = self
            .store
            .resolve(&params.input_image.to_string_lossy())?;
        params.input_image = resolved_input;

        let backend = self.select_backend(None).await?;
        let mut result = backend.upscale(&params).await?;
        let params_json = serde_json::to_value(&params).unwrap_or(serde_json::Value::Null);
        self.attach_artifacts(&mut result, "upscale", None, params_json);
        Ok(result)
    }

    /// Hash and copy each generated image into the artifact store, populating
    /// `result.artifacts` index-aligned with `result.images`. A failure on a
    /// single artifact is logged and skipped — generation already succeeded and
    /// returning the path-only result is still useful to local clients.
    fn attach_artifacts(
        &self,
        result: &mut GenerationResult,
        source_tool: &str,
        prompt: Option<String>,
        params_json: serde_json::Value,
    ) {
        for path in &result.images {
            if !path.exists() {
                // ComfyUI returns bare filenames; nothing to ingest from this side.
                continue;
            }
            let meta = MetaBuilder {
                source_tool,
                model: Some(result.model.clone()),
                seed: Some(result.seed),
                prompt: prompt.clone(),
                params: params_json.clone(),
            };
            match self.store.ingest_file(path, meta) {
                Ok(art) => result.artifacts.push(art),
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "artifact ingest failed");
                }
            }
        }
    }

    /// Resolve a model name against the registry.
    ///
    /// If found in the registry, returns (path_or_hf_id, architecture).
    /// If not found, passes the name through as-is — it could be a HuggingFace ID
    /// or a direct filesystem path that diffusers will resolve.
    /// Forward a raw RPC call to the first backend that supports worker_rpc.
    /// Used for non-generation operations (model search, download).
    pub async fn worker_rpc(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        for slot in &self.backends {
            match slot.backend.worker_rpc(method, params.clone()).await {
                Ok(result) => return Ok(result),
                Err(Error::BackendUnavailable(_)) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(Error::BackendUnavailable(
            "no backend supports worker_rpc".to_string(),
        ))
    }

    async fn resolve_model(&self, name: &str) -> (String, Option<crate::backend::Architecture>) {
        let reg = self.registry.read().await;
        if let Some(entry) = reg.get(name) {
            (entry.path.to_string_lossy().into_owned(), entry.architecture)
        } else {
            // Pass through — could be an HF ID like "black-forest-labs/FLUX.1-schnell"
            // or an absolute path. Let the backend handle resolution.
            (name.to_string(), None)
        }
    }

    async fn select_backend(
        &self,
        architecture: Option<crate::backend::Architecture>,
    ) -> Result<&dyn Backend> {
        for slot in &self.backends {
            if let Some(arch) = architecture {
                if !slot.backend.supported_architectures().contains(&arch) {
                    continue;
                }
            }

            if slot.healthy.load(Ordering::Relaxed) {
                return Ok(slot.backend.as_ref());
            }

            match slot.backend.health_check().await {
                Ok(s) if s.available => {
                    slot.healthy.store(true, Ordering::Relaxed);
                    return Ok(slot.backend.as_ref());
                }
                Ok(_) => {
                    warn!(backend = slot.backend.name(), "backend not available");
                }
                Err(e) => {
                    warn!(backend = slot.backend.name(), err = %e, "backend health check failed");
                }
            }
        }

        Err(Error::BackendUnavailable(
            architecture
                .map(|a| format!("no backend available for {}", a.as_str()))
                .unwrap_or_else(|| "no backends available".to_string()),
        ))
    }
}
