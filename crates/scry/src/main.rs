mod tools;

use scry_engine::{ComfyUiBackend, DiffusersBackend, Pipeline};
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use omegon_extension::{Extension, Result};
use serde_json::Value;
use tracing::info;

use crate::tools::dispatch_tool;

pub struct ScryExtension {
    pipeline: Arc<Pipeline>,
}

impl ScryExtension {
    pub fn new(extra_model_dirs: Vec<PathBuf>, output_dir: PathBuf, worker_script: PathBuf, python_bin: String) -> Self {
        let mut pipeline = Pipeline::new(&extra_model_dirs, output_dir.clone());

        // ComfyUI backend — tried first, gracefully absent if not running.
        let comfyui_url = std::env::var("COMFYUI_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8188".to_string());
        let comfyui = ComfyUiBackend::new(comfyui_url, output_dir);
        pipeline.add_backend(Box::new(comfyui));

        // Diffusers backend — always available as fallback.
        let diffusers = DiffusersBackend::new(worker_script, python_bin);
        pipeline.add_backend(Box::new(diffusers));

        Self {
            pipeline: Arc::new(pipeline),
        }
    }
}

#[async_trait]
impl Extension for ScryExtension {
    fn name(&self) -> &str {
        "scry"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    async fn handle_rpc(&self, method: &str, params: Value) -> Result<Value> {
        match method {
            // ── v2 handshake ──
            "initialize" => {
                Ok(serde_json::json!({
                    "protocol_version": 2,
                    "extension_info": {
                        "name": self.name(),
                        "version": self.version(),
                        "sdk_version": "0.16.0"
                    },
                    "capabilities": {
                        "tools": true, "widgets": true, "mind": false,
                        "vox": false, "resources": false, "prompts": false,
                        "sampling": false, "elicitation": false, "streaming": true
                    },
                    "tools": tools::tool_definitions()
                }))
            }

            // ── v2 tool dispatch ──
            "tools/call" => {
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(serde_json::json!({}));
                dispatch_tool(name, args, &self.pipeline).await
            }
            "execute_tool" => {
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = params.get("args").cloned().unwrap_or(serde_json::json!({}));
                dispatch_tool(name, args, &self.pipeline).await
            }

            "get_tools" | "tools/list" => Ok(tools::tool_definitions()),
            m if m.starts_with("execute_") => {
                let tool_name = &m["execute_".len()..];
                dispatch_tool(tool_name, params, &self.pipeline).await
            }
            "get_gallery" => {
                Ok(serde_json::json!({ "events": [] }))
            }
            "get_models" => {
                let models = self.pipeline.list_models(None).await;
                Ok(serde_json::to_value(models).unwrap_or_default())
            }
            _ => Err(omegon_extension::Error::method_not_found(method)),
        }
    }
}

fn resolve_dir(env_key: &str, default: &str) -> PathBuf {
    std::env::var(env_key)
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_or_default(default))
}

fn dirs_or_default(leaf: &str) -> PathBuf {
    let base = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join(".scry").join(leaf)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("scry=debug".parse().unwrap()),
        )
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str());

    // ~/.scry/models/ is our canonical location. HF cache, ComfyUI, A1111 are auto-discovered.
    // SCRY_EXTRA_MODEL_DIRS adds more paths (colon-separated).
    let scry_models_dir = resolve_dir("SCRY_MODELS_DIR", "models");
    let mut extra_model_dirs = vec![scry_models_dir.clone()];
    if let Ok(extra) = std::env::var("SCRY_EXTRA_MODEL_DIRS") {
        extra_model_dirs.extend(extra.split(':').map(PathBuf::from));
    }

    let output_dir = resolve_dir("SCRY_OUTPUT_DIR", "output");

    // Priority: SCRY_WORKER_SCRIPT env > relative to binary > cwd > ~/.scry/python/
    let worker_script = std::env::var("SCRY_WORKER_SCRIPT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."));

            let candidates = [
                exe_dir.join("worker.py"),               // installed: binary + worker.py side by side
                exe_dir.join("../../python/worker.py"),   // dev: target/release/../../python/
                exe_dir.join("../python/worker.py"),
                exe_dir.join("python/worker.py"),
                PathBuf::from("python/worker.py"),
            ];

            candidates
                .iter()
                .find(|p| p.exists())
                .cloned()
                .unwrap_or_else(|| dirs_or_default("python").join("worker.py"))
        });

    // Resolve the Python binary. Priority:
    // 1. SCRY_PYTHON env var (set by operator or omegon secrets)
    // 2. ~/.scry/venv/bin/python (created by install.sh)
    // 3. system python3
    let python_bin = std::env::var("SCRY_PYTHON").unwrap_or_else(|_| {
        let venv_python = dirs_or_default("venv").join("bin/python");
        if venv_python.exists() {
            venv_python.to_string_lossy().into_owned()
        } else {
            "python3".to_string()
        }
    });

    info!(
        scry_models = %scry_models_dir.display(),
        output_dir = %output_dir.display(),
        worker_script = %worker_script.display(),
        python = %python_bin,
        "starting scry extension"
    );

    let _ = std::fs::create_dir_all(&scry_models_dir);
    let _ = std::fs::create_dir_all(&output_dir);

    let ext = ScryExtension::new(extra_model_dirs, output_dir, worker_script, python_bin);

    match ext.pipeline.scan_models().await {
        Ok(n) => info!(n, "models scanned"),
        Err(e) => tracing::warn!(err = %e, "model scan failed"),
    }

    let result = match mode {
        Some("--mcp") => {
            // MCP server mode — compatible with Claude Code, Cursor, etc.
            omegon_extension::mcp_shim::serve_mcp(ext).await
        }
        Some("--help") | Some("help") | Some("-h") => {
            println!("scry — agentic local image generation for omegon");
            println!();
            println!("USAGE:");
            println!("  scry                Run as omegon extension (default, v2 protocol)");
            println!("  scry --rpc          Run as omegon extension (explicit)");
            println!("  scry --mcp          Run as MCP server (Claude Code, Cursor, etc.)");
            println!("  scry --help         Show this help");
            println!();
            println!("ENVIRONMENT:");
            println!("  COMFYUI_URL         ComfyUI server URL (default: http://127.0.0.1:8188)");
            println!("  SCRY_MODELS_DIR     Model directory (default: ~/.scry/models)");
            println!("  SCRY_OUTPUT_DIR     Output directory (default: ~/.scry/output)");
            return;
        }
        Some("--rpc") | _ => {
            // Default: omegon extension (v2 bidirectional protocol)
            omegon_extension::serve_v2(ext).await
        }
    };

    if let Err(e) = result {
        tracing::error!("scry serve failed: {e}");
        std::process::exit(1);
    }
}
