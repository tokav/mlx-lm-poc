/// Error types for MLXPack operations

use std::io;

#[derive(Debug, thiserror::Error)]
pub enum MLXPackError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("config.json not found: {0}")]
    ConfigNotFound(io::Error),

    #[error("tokenizer.json not found")]
    TokenizerNotFound,

    #[error("No safetensors files found (expected model.safetensors or model.safetensors.index.json)")]
    SafetensorsNotFound,

    #[error("Invalid model.safetensors.index.json format")]
    InvalidIndexFile,

    #[error("Invalid MLXPack file: {0}")]
    InvalidFormat(String),

    #[error("Unsupported MLXPack version: {0}")]
    UnsupportedVersion(u32),

    #[error("Safetensors file '{0}' not found in pack")]
    SafetensorsFileNotFound(String),
}

pub type Result<T> = std::result::Result<T, MLXPackError>;
