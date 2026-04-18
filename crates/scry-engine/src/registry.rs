use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::backend::Architecture;
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Checkpoint,
    Lora,
    Vae,
    Upscaler,
    TextEncoder,
    ControlNet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Safetensors,
    Gguf,
    Ckpt,
    Bin,
}

impl ModelFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "safetensors" => Some(Self::Safetensors),
            "gguf" => Some(Self::Gguf),
            "ckpt" | "pth" => Some(Self::Ckpt),
            "bin" => Some(Self::Bin),
            _ => None,
        }
    }
}

/// Where this model was discovered.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    /// HuggingFace Hub cache (~/.cache/huggingface/hub/).
    HuggingFace,
    /// ComfyUI models directory.
    ComfyUi,
    /// A1111 / Forge models directory.
    A1111,
    /// User-configured extra path.
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub kind: ModelKind,
    pub format: ModelFormat,
    pub architecture: Option<Architecture>,
    pub size_bytes: u64,
    pub source: ModelSource,
    #[serde(default)]
    pub tags: Vec<String>,
    /// LoRA-specific metadata, populated from safetensors header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_metadata: Option<LoraMetadata>,
}

/// Metadata extracted from a LoRA safetensors file header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraMetadata {
    /// Base model this LoRA was trained on (from ss_base_model_version).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>,
    /// Network rank/dimension (from ss_network_dim).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    /// Alpha value (from ss_network_alpha).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    /// Trigger phrase (from ss_training_comment or modelspec.trigger_phrase).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_phrase: Option<String>,
    /// Network module type (from ss_network_module).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_type: Option<String>,
    /// Architecture detected from tensor key names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_arch: Option<String>,
}

/// A directory to scan, with its expected content kind and source label.
#[derive(Debug, Clone)]
struct ScanDir {
    path: PathBuf,
    kind: ModelKind,
    source: ModelSource,
}

/// Scans all known model directories and catalogs what's available.
///
/// Respects the existing landscape:
/// - HuggingFace Hub cache (diffusers-format model directories)
/// - ComfyUI models/ subdirectories
/// - A1111/Forge models/ subdirectories
/// - User-configured extra paths
pub struct ModelRegistry {
    models: HashMap<String, ModelEntry>,
    scan_dirs: Vec<ScanDir>,
}

impl ModelRegistry {
    pub fn new(extra_dirs: &[PathBuf]) -> Self {
        let mut scan_dirs = Vec::new();

        // HuggingFace Hub cache — discover diffusers-format models.
        if let Some(hf_cache) = resolve_hf_cache() {
            scan_dirs.push(ScanDir {
                path: hf_cache,
                kind: ModelKind::Checkpoint,
                source: ModelSource::HuggingFace,
            });
        }

        // ComfyUI — scan common install locations.
        for comfy_root in find_comfyui_roots() {
            let models = comfy_root.join("models");
            for (subdir, kind) in COMFYUI_DIRS {
                scan_dirs.push(ScanDir {
                    path: models.join(subdir),
                    kind: *kind,
                    source: ModelSource::ComfyUi,
                });
            }
        }

        // A1111 / Forge — scan common install locations.
        for a1111_root in find_a1111_roots() {
            let models = a1111_root.join("models");
            for (subdir, kind) in A1111_DIRS {
                scan_dirs.push(ScanDir {
                    path: models.join(subdir),
                    kind: *kind,
                    source: ModelSource::A1111,
                });
            }
        }

        // User-configured extra directories — scan with ComfyUI convention.
        // If subdirs exist, scan them by kind. Also scan the root for loose files.
        for dir in extra_dirs {
            for (subdir, kind) in COMFYUI_DIRS {
                scan_dirs.push(ScanDir {
                    path: dir.join(subdir),
                    kind: *kind,
                    source: ModelSource::Custom,
                });
            }
            // Also scan root for loose checkpoint files.
            scan_dirs.push(ScanDir {
                path: dir.clone(),
                kind: ModelKind::Checkpoint,
                source: ModelSource::Custom,
            });
        }

        Self {
            models: HashMap::new(),
            scan_dirs,
        }
    }

    pub fn scan(&mut self) -> Result<usize> {
        self.models.clear();
        let mut count = 0;

        // Scan HuggingFace cache separately — it has a different structure.
        let hf_dirs: Vec<_> = self
            .scan_dirs
            .iter()
            .filter(|d| matches!(d.source, ModelSource::HuggingFace))
            .cloned()
            .collect();

        for scan_dir in &hf_dirs {
            count += self.scan_hf_cache(&scan_dir.path);
        }

        // Scan file-based directories (ComfyUI, A1111, custom).
        let file_dirs: Vec<_> = self
            .scan_dirs
            .iter()
            .filter(|d| !matches!(d.source, ModelSource::HuggingFace))
            .cloned()
            .collect();

        for scan_dir in &file_dirs {
            if !scan_dir.path.exists() {
                debug!(path = %scan_dir.path.display(), "scan dir does not exist, skipping");
                continue;
            }

            for entry in walkdir::WalkDir::new(&scan_dir.path)
                .max_depth(3)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let path = entry.path();
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or_default();

                let Some(format) = ModelFormat::from_extension(ext) else {
                    continue;
                };

                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);

                let mut architecture = guess_architecture(&name);
                let mut lora_metadata = None;

                // For LoRA safetensors, parse header metadata.
                if scan_dir.kind == ModelKind::Lora && format == ModelFormat::Safetensors {
                    if let Some(meta) = parse_lora_metadata(path) {
                        if architecture.is_none() {
                            if let Some(ref arch) = meta.detected_arch {
                                architecture = guess_architecture(arch);
                            }
                            if architecture.is_none() {
                                if let Some(ref base) = meta.base_model {
                                    architecture = guess_architecture(base);
                                }
                            }
                        }
                        lora_metadata = Some(meta);
                    }
                }

                let model = ModelEntry {
                    name: name.clone(),
                    path: path.to_path_buf(),
                    kind: scan_dir.kind,
                    format,
                    architecture,
                    size_bytes,
                    source: scan_dir.source.clone(),
                    tags: Vec::new(),
                    lora_metadata,
                };

                debug!(name = %model.name, kind = ?model.kind, source = ?model.source, "found model");

                // First-discovered wins — earlier scan dirs have priority.
                self.models.entry(name).or_insert(model);
                count += 1;
            }
        }

        info!(count, "model scan complete");
        Ok(count)
    }

    /// Scan the HuggingFace Hub cache for diffusers-format model directories.
    fn scan_hf_cache(&mut self, cache_dir: &Path) -> usize {
        if !cache_dir.exists() {
            return 0;
        }

        let mut count = 0;

        // Hub cache layout: hub/models--{org}--{name}/snapshots/{hash}/model_index.json
        let Ok(entries) = std::fs::read_dir(cache_dir) else {
            return 0;
        };

        for entry in entries.flatten() {
            let dir_name = entry.file_name();
            let dir_name = dir_name.to_string_lossy();

            if !dir_name.starts_with("models--") {
                continue;
            }

            // Convert models--org--name back to org/name.
            let hf_id = dir_name
                .strip_prefix("models--")
                .unwrap_or(&dir_name)
                .replace("--", "/");

            // Find the latest snapshot.
            let snapshots_dir = entry.path().join("snapshots");
            if !snapshots_dir.exists() {
                continue;
            }

            // Check if this is a diffusers model (has model_index.json in a snapshot).
            let is_diffusers = std::fs::read_dir(&snapshots_dir)
                .ok()
                .and_then(|mut entries| {
                    entries.find_map(|e| {
                        let e = e.ok()?;
                        if e.path().join("model_index.json").exists() {
                            Some(e.path())
                        } else {
                            None
                        }
                    })
                });

            if let Some(snapshot_path) = is_diffusers {
                let model = ModelEntry {
                    name: hf_id.clone(),
                    path: snapshot_path,
                    kind: ModelKind::Checkpoint,
                    format: ModelFormat::Safetensors,
                    architecture: guess_architecture(&hf_id),
                    size_bytes: 0, // Not meaningful for HF cache dirs.
                    source: ModelSource::HuggingFace,
                    tags: vec!["huggingface".to_string()],
                    lora_metadata: None,
                };

                debug!(name = %model.name, "found HF cached model");
                self.models.entry(hf_id).or_insert(model);
                count += 1;
            }
        }

        count
    }

    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.models.get(name)
    }

    pub fn list_owned(&self, kind: Option<ModelKind>) -> Vec<ModelEntry> {
        self.models
            .values()
            .filter(|m| kind.is_none() || kind == Some(m.kind))
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

// ── Directory conventions ────────────────────────────────────────────

/// ComfyUI subdirectory → ModelKind mapping.
const COMFYUI_DIRS: &[(&str, ModelKind)] = &[
    ("checkpoints", ModelKind::Checkpoint),
    ("unet", ModelKind::Checkpoint),
    ("loras", ModelKind::Lora),
    ("vae", ModelKind::Vae),
    ("upscale_models", ModelKind::Upscaler),
    ("clip", ModelKind::TextEncoder),
    ("controlnet", ModelKind::ControlNet),
];

/// A1111/Forge subdirectory → ModelKind mapping.
const A1111_DIRS: &[(&str, ModelKind)] = &[
    ("Stable-diffusion", ModelKind::Checkpoint),
    ("Lora", ModelKind::Lora),
    ("VAE", ModelKind::Vae),
    ("ESRGAN", ModelKind::Upscaler),
    ("RealESRGAN", ModelKind::Upscaler),
    ("ControlNet", ModelKind::ControlNet),
];

// ── Discovery helpers ────────────────────────────────────────────────

fn resolve_hf_cache() -> Option<PathBuf> {
    // HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > HF_HOME/hub > XDG_CACHE_HOME/huggingface/hub > ~/.cache/huggingface/hub
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return Some(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return Some(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(p).join("hub"));
    }
    if let Ok(p) = std::env::var("XDG_CACHE_HOME") {
        return Some(PathBuf::from(p).join("huggingface/hub"));
    }

    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// Find ComfyUI installations by checking common locations.
fn find_comfyui_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    // Check COMFYUI_PATH env var first.
    if let Ok(p) = std::env::var("COMFYUI_PATH") {
        roots.push(PathBuf::from(p));
    }

    if let Ok(home) = std::env::var("HOME") {
        let home = PathBuf::from(home);
        let candidates = [
            home.join("ComfyUI"),
            home.join("comfyui"),
            home.join(".local/share/ComfyUI"),
        ];
        for c in candidates {
            if c.join("models").exists() {
                roots.push(c);
            }
        }
    }

    roots
}

/// Find A1111/Forge installations by checking common locations.
fn find_a1111_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Ok(p) = std::env::var("A1111_PATH") {
        roots.push(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("SD_WEBUI_PATH") {
        roots.push(PathBuf::from(p));
    }

    if let Ok(home) = std::env::var("HOME") {
        let home = PathBuf::from(home);
        let candidates = [
            home.join("stable-diffusion-webui"),
            home.join("stable-diffusion-webui-forge"),
            home.join(".local/share/stable-diffusion-webui"),
        ];
        for c in candidates {
            if c.join("models").exists() {
                roots.push(c);
            }
        }
    }

    roots
}

fn guess_architecture(name: &str) -> Option<Architecture> {
    let lower = name.to_lowercase();

    if lower.contains("flux2") || lower.contains("flux-2") || lower.contains("flux.2") {
        return Some(Architecture::Flux2);
    }
    if lower.contains("flux") {
        return Some(Architecture::Flux);
    }
    if lower.contains("sd3") || lower.contains("sd-3") {
        return Some(Architecture::Sd3);
    }
    if lower.contains("sdxl") || lower.contains("sd_xl") {
        return Some(Architecture::Sdxl);
    }
    if lower.contains("sd-2") || lower.contains("sd2") || lower.contains("v2-1") {
        return Some(Architecture::Sd21);
    }
    if lower.contains("sd-1") || lower.contains("sd1") || lower.contains("v1-5") {
        return Some(Architecture::Sd15);
    }

    None
}

// ── LoRA metadata parsing ────────────────────────────────────────────

/// Parse LoRA metadata from a safetensors file header.
/// This only reads the JSON header (first few KB), not the full tensor data.
fn parse_lora_metadata(path: &Path) -> Option<LoraMetadata> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;

    // safetensors format: first 8 bytes = little-endian u64 header size,
    // then header_size bytes of JSON metadata.
    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf).ok()?;
    let header_size = u64::from_le_bytes(size_buf) as usize;

    // Sanity check — headers shouldn't be > 10MB.
    if header_size > 10_000_000 {
        return None;
    }

    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf).ok()?;

    let header: serde_json::Value = serde_json::from_slice(&header_buf).ok()?;
    let meta = header.get("__metadata__")?.as_object()?;

    let get_str = |key: &str| -> Option<String> {
        meta.get(key)?.as_str().map(|s| s.to_string())
    };

    let base_model = get_str("ss_base_model_version");
    let rank = get_str("ss_network_dim").and_then(|s| s.parse::<u32>().ok());
    let alpha = get_str("ss_network_alpha").and_then(|s| s.parse::<f32>().ok());
    let network_type = get_str("ss_network_module");

    let trigger_phrase = get_str("modelspec.trigger_phrase")
        .or_else(|| get_str("ss_training_comment"));

    // Detect architecture from tensor key names.
    let detected_arch = detect_arch_from_keys(header.as_object()?);

    Some(LoraMetadata {
        base_model,
        rank,
        alpha,
        trigger_phrase,
        network_type,
        detected_arch,
    })
}

/// Detect the target model architecture from safetensors tensor key names.
fn detect_arch_from_keys(header: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    let has_key_containing = |pattern: &str| -> bool {
        header.keys().any(|k| k.contains(pattern))
    };

    // FLUX: uses transformer blocks with DiT naming.
    if has_key_containing("transformer.single_transformer_blocks") {
        return Some("flux".to_string());
    }

    // SDXL: two text encoders.
    if has_key_containing("lora_te1_") && has_key_containing("lora_te2_") {
        return Some("sdxl".to_string());
    }

    // SD3: different block naming.
    if has_key_containing("transformer.blocks.") && !has_key_containing("lora_unet_") {
        return Some("sd3".to_string());
    }

    // SD 1.5: single text encoder + UNet.
    if has_key_containing("lora_unet_") && has_key_containing("lora_te_") {
        return Some("sd15".to_string());
    }

    // UNet-based but no text encoder info — could be SD 1.5 or 2.x.
    if has_key_containing("lora_unet_") {
        return Some("sd".to_string());
    }

    None
}
