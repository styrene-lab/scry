use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::backend::{Architecture, Backend, BackendStatus};
use crate::params::{GenerateParams, GenerationResult, Img2ImgParams, UpscaleParams};
use crate::{Error, Result};

/// Shared secret store. Populated by the omegon `bootstrap_secrets` RPC (or seeded
/// from process env in --mcp mode). Read at worker spawn time and injected as env
/// vars on the worker subprocess only — never persisted, never echoed in logs.
pub type SecretStore = Arc<Mutex<HashMap<String, String>>>;

/// Backend that delegates to a long-running Python diffusers worker process.
/// Communicates via line-delimited JSON over stdin/stdout.
pub struct DiffusersBackend {
    worker_script: PathBuf,
    python_bin: String,
    process: Mutex<Option<WorkerProcess>>,
    request_id: AtomicU64,
    secrets: SecretStore,
}

struct WorkerProcess {
    #[allow(dead_code)] // Held for process lifetime; kill_on_drop fires when dropped.
    child: Child,
    stdin: tokio::process::ChildStdin,
    stdout: BufReader<tokio::process::ChildStdout>,
}

impl DiffusersBackend {
    pub fn new(worker_script: PathBuf, python_bin: String) -> Self {
        Self::with_secrets(worker_script, python_bin, Arc::new(Mutex::new(HashMap::new())))
    }

    pub fn with_secrets(worker_script: PathBuf, python_bin: String, secrets: SecretStore) -> Self {
        Self {
            worker_script,
            python_bin,
            process: Mutex::new(None),
            request_id: AtomicU64::new(1),
            secrets,
        }
    }

    async fn ensure_worker(&self) -> Result<()> {
        let mut proc = self.process.lock().await;
        if proc.is_some() {
            return Ok(());
        }

        info!(
            script = %self.worker_script.display(),
            python = %self.python_bin,
            "spawning diffusers worker"
        );

        let mut cmd = Command::new(&self.python_bin);
        cmd.arg("-u")
            .arg(&self.worker_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // Inject secrets as env vars on the worker only. huggingface_hub reads
        // HF_TOKEN automatically; CIVITAI_TOKEN is read by our search_models impl.
        // Logged by name only — values never appear in stderr.
        let secrets = self.secrets.lock().await;
        if !secrets.is_empty() {
            let names: Vec<&str> = secrets.keys().map(|s| s.as_str()).collect();
            debug!(secrets = ?names, "injecting secrets into worker env");
            for (k, v) in secrets.iter() {
                cmd.env(k, v);
            }
        }
        drop(secrets);

        let mut child = cmd
            .spawn()
            .map_err(|e| Error::BackendUnavailable(format!("failed to spawn python worker: {e}")))?;

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        let stderr = child.stderr.take().unwrap();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let trimmed = line.trim_end();
                        if !trimmed.is_empty() {
                            debug!(target: "scry::diffusers::worker", "{trimmed}");
                        }
                    }
                }
            }
        });

        *proc = Some(WorkerProcess {
            child,
            stdin,
            stdout,
        });

        info!("diffusers worker spawned");
        Ok(())
    }

    /// Send a raw RPC call to the Python worker. Used for non-generation operations
    /// like model search/download.
    pub async fn rpc(&self, method: &str, params: Value) -> Result<Value> {
        self.ensure_worker().await?;

        let id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let request = json!({
            "id": id.to_string(),
            "method": method,
            "params": params,
        });

        let mut proc = self.process.lock().await;
        let worker = proc.as_mut().ok_or_else(|| {
            Error::BackendUnavailable("worker process not running".to_string())
        })?;

        let mut request_line = serde_json::to_string(&request)
            .map_err(|e| Error::GenerationFailed(format!("failed to serialize request: {e}")))?;
        request_line.push('\n');

        // If the cached worker died (e.g. crashed, oom-killed, or killed externally),
        // its pipes are broken. Drop the stale handle so the next call respawns.
        if let Err(e) = worker.stdin.write_all(request_line.as_bytes()).await {
            *proc = None;
            return Err(Error::BackendUnavailable(format!(
                "failed to write to worker stdin: {e} (cached worker dropped; next call will respawn)"
            )));
        }
        worker.stdin.flush().await.ok();

        let mut response_line = String::new();
        if let Err(e) = worker.stdout.read_line(&mut response_line).await {
            *proc = None;
            return Err(Error::BackendUnavailable(format!(
                "failed to read from worker stdout: {e} (cached worker dropped; next call will respawn)"
            )));
        }

        if response_line.is_empty() {
            *proc = None;
            return Err(Error::BackendUnavailable(
                "worker process exited unexpectedly".to_string(),
            ));
        }

        let response: Value = serde_json::from_str(&response_line)
            .map_err(|e| Error::GenerationFailed(format!("invalid worker response: {e}")))?;

        if let Some(err) = response.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(Error::GenerationFailed(msg.to_string()));
        }

        response
            .get("result")
            .cloned()
            .ok_or_else(|| Error::GenerationFailed("missing result in worker response".to_string()))
    }

    fn loras_to_json(params: &GenerateParams) -> Vec<Value> {
        params
            .loras
            .iter()
            .map(|l| json!({ "path": l.model, "weight": l.weight }))
            .collect()
    }

    fn output_dir_str(dir: Option<&Path>) -> String {
        dir.map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string())
    }

    fn parse_result(result: Value, model: &str) -> Result<GenerationResult> {
        let images: Vec<PathBuf> = result
            .get("images")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(PathBuf::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(GenerationResult {
            seed: result.get("seed").and_then(|v| v.as_i64()).unwrap_or(-1),
            elapsed_ms: result.get("elapsed_ms").and_then(|v| v.as_u64()).unwrap_or(0),
            model: model.to_string(),
            images,
            artifacts: Vec::new(),
        })
    }
}

#[async_trait]
impl Backend for DiffusersBackend {
    fn name(&self) -> &str {
        "diffusers"
    }

    fn supported_architectures(&self) -> &[Architecture] {
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
        match self.rpc("health", json!({})).await {
            Ok(result) => Ok(BackendStatus {
                available: result.get("available").and_then(|v| v.as_bool()).unwrap_or(false),
                name: self.name().to_string(),
                version: result.get("torch_version").and_then(|v| v.as_str()).map(String::from),
                device: result.get("device").and_then(|v| v.as_str()).map(String::from),
            }),
            Err(e) => {
                warn!(err = %e, "diffusers health check failed");
                Ok(BackendStatus {
                    available: false,
                    name: self.name().to_string(),
                    version: None,
                    device: None,
                })
            }
        }
    }

    async fn generate(&self, params: &GenerateParams) -> Result<GenerationResult> {
        let rpc_params = json!({
            "model_path": params.model,
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "width": params.width,
            "height": params.height,
            "steps": params.steps,
            "cfg_scale": params.cfg_scale,
            "sampler": params.sampler,
            "seed": params.seed,
            "batch_size": params.batch_size,
            "vae": params.vae,
            "loras": Self::loras_to_json(params),
            "output_dir": Self::output_dir_str(params.output_dir.as_deref()),
        });

        let result = self.rpc("generate", rpc_params).await?;
        Self::parse_result(result, &params.model)
    }

    async fn img2img(&self, params: &Img2ImgParams) -> Result<GenerationResult> {
        let rpc_params = json!({
            "model_path": params.base.model,
            "prompt": params.base.prompt,
            "negative_prompt": params.base.negative_prompt,
            "input_image": params.input_image.to_string_lossy(),
            "strength": params.strength,
            "steps": params.base.steps,
            "cfg_scale": params.base.cfg_scale,
            "sampler": params.base.sampler,
            "seed": params.base.seed,
            "batch_size": params.base.batch_size,
            "vae": params.base.vae,
            "loras": Self::loras_to_json(&params.base),
            "output_dir": Self::output_dir_str(params.base.output_dir.as_deref()),
        });

        let result = self.rpc("img2img", rpc_params).await?;
        Self::parse_result(result, &params.base.model)
    }

    async fn upscale(&self, params: &UpscaleParams) -> Result<GenerationResult> {
        let rpc_params = json!({
            "input_image": params.input_image.to_string_lossy(),
            "factor": params.factor,
            "upscaler_path": params.upscaler.as_deref().unwrap_or(""),
            "output_dir": Self::output_dir_str(params.output_dir.as_deref()),
        });

        let result = self.rpc("upscale", rpc_params).await?;
        Self::parse_result(result, "upscaler")
    }

    async fn worker_rpc(&self, method: &str, params: Value) -> Result<Value> {
        self.rpc(method, params).await
    }
}

impl Drop for DiffusersBackend {
    fn drop(&mut self) {
        let proc = self.process.get_mut();
        if proc.is_some() {
            *proc = None;
            info!("diffusers worker dropped");
        }
    }
}
