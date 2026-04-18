pub mod backend;
pub mod backends;
pub mod error;
pub mod params;
pub mod pipeline;
pub mod registry;
pub mod workflow;
pub mod workflow_templates;

pub use backends::{ComfyUiBackend, DiffusersBackend};
pub use error::{Error, Result};
pub use params::{GenerateParams, GenerationResult, Img2ImgParams, UpscaleParams};
pub use pipeline::Pipeline;
pub use registry::{ModelEntry, ModelKind, ModelRegistry};
pub use workflow::Workflow;
pub use workflow_templates::{WorkflowParams, flux_txt2img, sd_txt2img};
