use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use crate::quantized::load_quantized_safetensors;

pub mod llama;
pub mod mistral;
pub mod qwen3;

// ============================================================================
// Common Model Configuration Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeScaling {
    pub factor: f32,
    pub high_freq_factor: f32,
    pub low_freq_factor: f32,
    pub original_max_position_embeddings: i32,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

/// Unified model configuration that works for Mistral, Qwen3, and Llama models
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelArgs {
    // Model architecture type (e.g., ["LlamaForCausalLM"])
    #[serde(default)]
    pub architectures: Option<Vec<String>>,

    // Core model dimensions - common to all models
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,

    // RoPE configuration
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    // Head dimension (explicit for Qwen3/Llama, computed for Mistral)
    #[serde(default)]
    pub head_dim: Option<i32>,

    // Bias configuration
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,

    // Weight tying
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // Optional fields
    #[serde(default)]
    pub max_position_embeddings: Option<i32>,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub bos_token_id: Option<Value>,
    #[serde(default)]
    pub eos_token_id: Option<Value>,
    #[serde(default)]
    pub attention_dropout: Option<f32>,
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
    #[serde(default)]
    pub max_window_layers: Option<i32>,

    // Quantization configuration
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl ModelArgs {
    pub const DEFAULT_ROPE_THETA: f32 = 10000.0;

    /// Get model dimension (hidden_size)
    pub fn dim(&self) -> i32 {
        self.hidden_size
    }

    /// Get number of layers (num_hidden_layers)
    pub fn n_layers(&self) -> i32 {
        self.num_hidden_layers
    }

    /// Get hidden dimension for FFN (intermediate_size)
    pub fn hidden_dim(&self) -> i32 {
        self.intermediate_size
    }

    /// Get number of attention heads (num_attention_heads)
    pub fn n_heads(&self) -> i32 {
        self.num_attention_heads
    }

    /// Get number of key-value heads (num_key_value_heads)
    pub fn n_kv_heads(&self) -> i32 {
        self.num_key_value_heads
    }

    /// Get normalization epsilon (rms_norm_eps)
    pub fn norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    /// Get head dimension (explicit or computed)
    pub fn head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Get maximum context size from model configuration
    pub fn max_context_size(&self) -> i32 {
        self.max_position_embeddings.unwrap_or(2048)
    }
}

// ============================================================================
// Common Helper Functions
// ============================================================================

/// Load model configuration from config.json
fn load_model_config(model_path: &Path) -> Result<ModelArgs, ModelError> {
    let config_path = model_path.join("config.json");
    let file = std::fs::File::open(config_path)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;
    Ok(model_args)
}

/// Load weights from safetensors files (handles both single file and sharded models)
fn load_weights<T: mlx_rs::module::ModuleParametersExt>(
    model: &mut T,
    model_path: &Path,
    model_args: &ModelArgs,
) -> Result<(), ModelError> {
    let weights_index = model_path.join("model.safetensors.index.json");

    if weights_index.exists() {
        // Sharded model
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

        for weight_file in weight_files {
            let weights_filename = model_path.join(weight_file);
            if model_args.quantization.is_some() {
                load_quantized_safetensors(model, &weights_filename)?;
            } else {
                model.load_safetensors(weights_filename)?;
            }
        }
    } else {
        let weights_filename = model_path.join("model.safetensors");
        if model_args.quantization.is_some() {
            load_quantized_safetensors(model, &weights_filename)?;
        } else {
            model.load_safetensors(weights_filename)?;
        }
    }

    Ok(())
}

// ============================================================================
// Model Input/Output Types
// ============================================================================

use crate::generate::kv_cache::TokensCache;

pub struct ModelInput<'a> {
    pub inputs: &'a Array,
    pub cache: TokensCache,
}

pub struct ModelOutput {
    pub logits: Array,
}

// ============================================================================
// Model Error Types
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Invalid vocab size: {0}")]
    InvalidVocabSize(i32),

    #[error("MLX exception: {0}")]
    MlxException(#[from] mlx_rs::error::Exception),

    #[error("MLX IO error: {0}")]
    MlxIoError(#[from] mlx_rs::error::IoError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensorsError(#[from] safetensors::SafeTensorError),

    #[error("Unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),
}

// ============================================================================
// LanguageModel Trait
// ============================================================================

pub trait LanguageModel {
    fn forward(&mut self, input: ModelInput<'_>) -> Result<ModelOutput, ModelError>;
    fn max_context_size(&self) -> usize;
}

impl LanguageModel for Box<dyn LanguageModel> {
    fn forward(&mut self, input: ModelInput<'_>) -> Result<ModelOutput, ModelError> {
        (**self).forward(input)
    }

    fn max_context_size(&self) -> usize {
        (**self).max_context_size()
    }
}

pub fn from_pretrained(model_path: &Path) -> Result<Box<dyn LanguageModel>, ModelError> {
    // Load model configuration
    let model_args = load_model_config(model_path)?;

    // Read config.json to determine architecture
    let config_path = model_path.join("config.json");
    let config_content = std::fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;

    // Get the architecture type
    let architectures = config["architectures"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            ModelError::UnsupportedArchitecture("No architecture specified".to_string())
        })?;

    let model: Box<dyn LanguageModel> = match architectures {
        "MistralForCausalLM" => {
            let mut model = mistral::Model::new(&model_args)?;
            if let Some(quant_config) = &model_args.quantization {
                eprintln!(
                    "[INFO] Pre-quantized model detected ({}bit, group_size={})",
                    quant_config.bits, quant_config.group_size
                );
                eprintln!("[INFO] Quantizing model structure before loading weights");
                model = mlx_rs::nn::quantize(
                    model,
                    Some(quant_config.group_size),
                    Some(quant_config.bits),
                )?;
            }
            load_weights(&mut model, model_path, &model_args)?;
            Box::new(model)
        }
        "Qwen3ForCausalLM" => {
            let mut model = qwen3::Model::new(&model_args)?;
            if let Some(quant_config) = &model_args.quantization {
                eprintln!(
                    "[INFO] Pre-quantized model detected ({}bit, group_size={})",
                    quant_config.bits, quant_config.group_size
                );
                eprintln!("[INFO] Quantizing model structure before loading weights");
                model = mlx_rs::nn::quantize(
                    model,
                    Some(quant_config.group_size),
                    Some(quant_config.bits),
                )?;
            }
            load_weights(&mut model, model_path, &model_args)?;
            Box::new(model)
        }
        "LlamaForCausalLM" => {
            let mut model = llama::Model::new(&model_args)?;
            if let Some(quant_config) = &model_args.quantization {
                eprintln!(
                    "[INFO] Pre-quantized model detected ({}bit, group_size={})",
                    quant_config.bits, quant_config.group_size
                );
                eprintln!("[INFO] Quantizing model structure before loading weights");
                model = mlx_rs::nn::quantize(
                    model,
                    Some(quant_config.group_size),
                    Some(quant_config.bits),
                )?;
            }
            load_weights(&mut model, model_path, &model_args)?;
            Box::new(model)
        }
        arch => return Err(ModelError::UnsupportedArchitecture(arch.to_string()).into()),
    };
    Ok(model)
}
