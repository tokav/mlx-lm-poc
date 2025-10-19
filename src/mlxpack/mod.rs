/// MLXPack: Single-file packaging format for MLX models
///
/// This module provides serialization/deserialization for packaging HuggingFace
/// format models (config, tokenizer, safetensors) into a single file while
/// preserving MLX quantization and avoiding disk space duplication.
///
/// Format structure:
/// ```
/// [Header - 16 bytes]
///   - magic: "MLXP" (4 bytes)
///   - version: u32 (4 bytes)
///   - metadata_size: u64 (8 bytes)
///
/// [Metadata - JSON]
///   {
///     "config": {...},           // config.json contents
///     "tokenizer_config": {...}, // tokenizer_config.json contents (optional)
///     "generation_config": {...},// generation_config.json contents (optional)
///     "chat_template": "...",    // chat_template.jinja contents (optional)
///     "safetensors_files": [     // list of safetensors file entries
///       {
///         "name": "model.safetensors",
///         "offset": 12345,
///         "size": 67890
///       },
///       ...
///     ]
///   }
///
/// [Safetensors Files - Binary]
///   - Concatenated safetensors file data
/// ```

use std::fs::File;
use std::path::Path;
use serde::{Deserialize, Serialize};

mod writer;
mod reader;
mod error;
mod loader;

pub use writer::MLXPackWriter;
pub use reader::MLXPackReader;
pub use error::{MLXPackError, Result};
pub use loader::load_from_mlxpack;

/// Magic number for MLXPack files: "MLXP"
pub const MAGIC: [u8; 4] = *b"MLXP";

/// Current format version
pub const VERSION: u32 = 1;

/// Header size in bytes (magic + version + metadata_size)
pub const HEADER_SIZE: usize = 16;

/// Metadata for a single safetensors file embedded in the pack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsFileEntry {
    /// Original filename (e.g., "model.safetensors" or "model-00001-of-00003.safetensors")
    pub name: String,
    /// Offset from start of safetensors data section
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
}

/// Metadata section containing all JSON configs and file layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackMetadata {
    /// config.json contents (parsed as ModelArgs)
    pub config: crate::models::ModelArgs,

    /// tokenizer.json contents as raw string (required for tokenizers::Tokenizer::from_str)
    pub tokenizer_json: String,

    /// tokenizer_config.json contents (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_config: Option<crate::tokenizer::TokenizerConfig>,

    /// generation_config.json contents (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<crate::tokenizer::GenerationConfig>,

    /// chat_template.jinja contents (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,

    /// List of embedded safetensors files with their offsets
    pub safetensors_files: Vec<SafetensorsFileEntry>,
}

impl PackMetadata {
    /// Create metadata from a HuggingFace model directory
    pub fn from_model_dir(model_path: &Path) -> Result<Self> {
        // Load required config.json
        let config_path = model_path.join("config.json");
        let config: crate::models::ModelArgs = serde_json::from_reader(
            File::open(&config_path).map_err(|e| MLXPackError::ConfigNotFound(e))?
        )?;

        // Load required tokenizer.json
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer_json = std::fs::read_to_string(&tokenizer_path)
            .map_err(|_| MLXPackError::TokenizerNotFound)?;

        // Load optional tokenizer_config.json
        let tokenizer_config_path = model_path.join("tokenizer_config.json");
        let tokenizer_config = if tokenizer_config_path.exists() {
            Some(serde_json::from_reader(File::open(&tokenizer_config_path)?)?)
        } else {
            None
        };

        // Load optional generation_config.json
        let generation_config_path = model_path.join("generation_config.json");
        let generation_config = if generation_config_path.exists() {
            Some(serde_json::from_reader(File::open(&generation_config_path)?)?)
        } else {
            None
        };

        // Load optional chat_template.jinja
        let chat_template_path = model_path.join("chat_template.jinja");
        let chat_template = if chat_template_path.exists() {
            Some(std::fs::read_to_string(&chat_template_path)?)
        } else {
            None
        };

        // Detect safetensors files (sharded or single)
        let safetensors_files = Self::detect_safetensors_files(model_path)?;

        Ok(Self {
            config,
            tokenizer_json,
            tokenizer_config,
            generation_config,
            chat_template,
            safetensors_files,
        })
    }

    /// Detect all safetensors files in the model directory
    /// Handles both single file (model.safetensors) and sharded models (model.safetensors.index.json)
    fn detect_safetensors_files(model_path: &Path) -> Result<Vec<SafetensorsFileEntry>> {
        let index_path = model_path.join("model.safetensors.index.json");

        if index_path.exists() {
            // Sharded model: read index to get all shard filenames
            let index_content = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_content)?;

            // Extract unique filenames from weight_map
            let weight_map = index["weight_map"]
                .as_object()
                .ok_or(MLXPackError::InvalidIndexFile)?;

            let mut filenames: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect();

            // Remove duplicates and sort for consistent ordering
            filenames.sort();
            filenames.dedup();

            // Create entries (offsets will be calculated during packing)
            let entries: Result<Vec<SafetensorsFileEntry>> = filenames
                .into_iter()
                .map(|name| {
                    let path = model_path.join(&name);
                    let metadata = std::fs::metadata(&path)?;
                    Ok(SafetensorsFileEntry {
                        name,
                        offset: 0, // Will be set during packing
                        size: metadata.len(),
                    })
                })
                .collect();

            entries
        } else {
            // Single file model
            let single_path = model_path.join("model.safetensors");
            if !single_path.exists() {
                return Err(MLXPackError::SafetensorsNotFound);
            }

            let metadata = std::fs::metadata(&single_path)?;
            Ok(vec![SafetensorsFileEntry {
                name: "model.safetensors".to_string(),
                offset: 0, // Will be set during packing
                size: metadata.len(),
            }])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_constants() {
        assert_eq!(MAGIC, *b"MLXP");
        assert_eq!(VERSION, 1);
        assert_eq!(HEADER_SIZE, 16);
    }
}
