//! Content-addressed artifact store.
//!
//! Layout: `<root>/<sha[0:2]>/<sha>.<ext>` for the bytes, with a
//! `<sha>.json` sidecar carrying tool-level metadata. The sidecar is what
//! a remote `get_artifact` lookup returns alongside the inline bytes —
//! it has to survive after the original tool result is gone from a
//! conversation.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Result;

/// One stored artifact, as surfaced in `details.artifacts[]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// sha256 hex of the file bytes.
    pub id: String,
    /// `scry://artifact/<sha256>` — the form refine/upscale accept.
    pub uri: String,
    /// Local filesystem path inside the store.
    pub path: PathBuf,
    pub media_type: String,
    pub bytes: u64,
}

/// Sidecar metadata persisted next to each artifact as `<sha>.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMeta {
    pub id: String,
    pub media_type: String,
    pub bytes: u64,
    /// Tool that produced this artifact: "generate" | "refine" | "upscale".
    pub source_tool: String,
    pub model: Option<String>,
    pub seed: Option<i64>,
    pub prompt: Option<String>,
    /// ISO-8601 UTC.
    pub created_at: String,
    /// Free-form parameter snapshot — the params struct serialized to JSON.
    #[serde(default)]
    pub params: serde_json::Value,
}

pub const URI_SCHEME: &str = "scry://artifact/";

#[derive(Debug, Clone)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Hash bytes already on disk, copy them into the store, and write the sidecar.
    /// If the artifact already exists (same sha) the existing copy is returned and
    /// the sidecar is left alone — content-addressed dedupe.
    pub fn ingest_file(&self, src: &Path, meta_builder: MetaBuilder<'_>) -> Result<Artifact> {
        let bytes = std::fs::read(src)?;
        let id = sha256_hex(&bytes);
        let media_type = media_type_for(src).to_string();
        let ext = ext_for(&media_type);

        let dir = self.root.join(&id[..2]);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{id}.{ext}"));
        let sidecar = dir.join(format!("{id}.json"));

        if !path.exists() {
            std::fs::write(&path, &bytes)?;
        }
        if !sidecar.exists() {
            let meta = meta_builder.build(id.clone(), media_type.clone(), bytes.len() as u64);
            let json = serde_json::to_vec_pretty(&meta).map_err(|e| {
                crate::Error::Other(anyhow::anyhow!("sidecar serialize: {e}"))
            })?;
            std::fs::write(&sidecar, json)?;
        }

        Ok(Artifact {
            id: id.clone(),
            uri: format!("{URI_SCHEME}{id}"),
            path,
            media_type,
            bytes: bytes.len() as u64,
        })
    }

    /// Look up an artifact by sha256 hex. Returns the on-disk path and parsed sidecar.
    pub fn get(&self, id: &str) -> Result<(PathBuf, ArtifactMeta)> {
        let dir = self.root.join(&id[..2]);
        let sidecar = dir.join(format!("{id}.json"));
        let meta_bytes = std::fs::read(&sidecar)?;
        let meta: ArtifactMeta = serde_json::from_slice(&meta_bytes)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("sidecar parse: {e}")))?;
        let ext = ext_for(&meta.media_type);
        let path = dir.join(format!("{id}.{ext}"));
        if !path.exists() {
            return Err(crate::Error::Other(anyhow::anyhow!(
                "artifact bytes missing for {id}"
            )));
        }
        Ok((path, meta))
    }

    /// Resolve either a `scry://artifact/<sha>` URI or a plain filesystem path
    /// to an on-disk path. Used by refine/upscale to accept artifact IDs.
    pub fn resolve(&self, input: &str) -> Result<PathBuf> {
        if let Some(id) = input.strip_prefix(URI_SCHEME) {
            let (path, _) = self.get(id)?;
            Ok(path)
        } else {
            Ok(PathBuf::from(input))
        }
    }
}

/// Builder for the sidecar metadata. Lives next to ingest_file so callers
/// don't have to know the id/bytes/media_type ahead of time.
pub struct MetaBuilder<'a> {
    pub source_tool: &'a str,
    pub model: Option<String>,
    pub seed: Option<i64>,
    pub prompt: Option<String>,
    pub params: serde_json::Value,
}

impl<'a> MetaBuilder<'a> {
    fn build(self, id: String, media_type: String, bytes: u64) -> ArtifactMeta {
        ArtifactMeta {
            id,
            media_type,
            bytes,
            source_tool: self.source_tool.to_string(),
            model: self.model,
            seed: self.seed,
            prompt: self.prompt,
            created_at: now_iso8601(),
            params: self.params,
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
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
        _ => "image/png",
    }
}

fn ext_for(media_type: &str) -> &'static str {
    match media_type {
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "png",
    }
}

fn now_iso8601() -> String {
    // Good-enough ISO-8601 without pulling chrono. SystemTime → seconds since
    // epoch, then formatted manually as UTC.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    format_utc(secs)
}

fn format_utc(secs: i64) -> String {
    // Civil from days, Howard Hinnant's algorithm. Avoids chrono.
    let days = secs.div_euclid(86_400);
    let time = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let hh = time / 3600;
    let mm = (time % 3600) / 60;
    let ss = time % 60;
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip() {
        let tmp = tempdir();
        let store = ArtifactStore::new(tmp.clone());

        // fake an existing PNG
        let src = tmp.join("input.png");
        std::fs::write(&src, b"\x89PNGfakebytes").unwrap();

        let art = store
            .ingest_file(
                &src,
                MetaBuilder {
                    source_tool: "generate",
                    model: Some("flux-dev".into()),
                    seed: Some(42),
                    prompt: Some("hi".into()),
                    params: json!({"width": 1024}),
                },
            )
            .unwrap();

        assert_eq!(art.id.len(), 64);
        assert!(art.uri.starts_with(URI_SCHEME));
        assert!(art.path.exists());

        let (path, meta) = store.get(&art.id).unwrap();
        assert_eq!(path, art.path);
        assert_eq!(meta.model.as_deref(), Some("flux-dev"));
        assert_eq!(meta.seed, Some(42));

        // resolve via URI
        let resolved = store.resolve(&art.uri).unwrap();
        assert_eq!(resolved, art.path);

        // resolve plain path passes through
        let plain = store.resolve("/tmp/whatever.png").unwrap();
        assert_eq!(plain, PathBuf::from("/tmp/whatever.png"));

        // dedupe: ingesting the same bytes returns the same id and doesn't error
        let again = store
            .ingest_file(
                &src,
                MetaBuilder {
                    source_tool: "generate",
                    model: None,
                    seed: None,
                    prompt: None,
                    params: serde_json::Value::Null,
                },
            )
            .unwrap();
        assert_eq!(again.id, art.id);
    }

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "scry-store-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn iso_format_basic() {
        // 2026-04-26T00:00:00Z
        let secs = 1_777_507_200;
        assert_eq!(format_utc(secs), "2026-04-30T00:00:00Z");
    }
}
