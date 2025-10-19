use std::path::Path;
use crate::models::{LanguageModel, ModelError};
use crate::tokenizer::{Tokenizer, TokenizerError};

pub mod models;
pub mod tokenizer;
pub mod generate;
pub mod quantized;
pub mod mlxpack;
pub mod mlx_rs_patch;

#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] TokenizerError),

    #[error("Model error: {0}")]
    ModelError(#[from] ModelError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("MLX exception: {0}")]
    MlxException(#[from] mlx_rs::error::Exception),

    #[error("MLXPack error: {0}")]
    MLXPackError(#[from] mlxpack::MLXPackError),
}

pub fn load(model_path: &Path, verbose: bool) -> Result<(Box<dyn LanguageModel>, Tokenizer), LoaderError> {
    if verbose {
        eprintln!("[INFO] Loading from folder: {:?}", model_path);
    }
    let tokenizer = Tokenizer::from_model_dir(model_path)?;
    if verbose {
        if let (Some(bos), Some(bos_id)) = (tokenizer.bos_token(), tokenizer.bos_token_id()) {
            eprintln!("[DEBUG] BOS token: {} (id={})", bos, bos_id);
        }
        if let (Some(eos), Some(eos_id)) = (tokenizer.eos_token(), tokenizer.eos_token_id()) {
            eprintln!("[DEBUG] EOS token: {} (id={})", eos, eos_id);
        }
        if let Some(pad) = tokenizer.pad_token() {
            eprintln!("[DEBUG] PAD token: {}", pad);
        }
        eprintln!("[DEBUG] Add BOS token: {}", tokenizer.add_bos_token());
        eprintln!("[DEBUG] Add EOS token: {}", tokenizer.add_eos_token());
        eprintln!("[DEBUG] Add prefix space: {}", tokenizer.add_prefix_space());
        eprintln!("[DEBUG] Clean up tokenization spaces: {}", tokenizer.clean_up_tokenization_spaces());
        eprintln!("[DEBUG] Spaces between special tokens: {}", tokenizer.spaces_between_special_tokens());
        eprintln!("[DEBUG] Use default system prompt: {}", tokenizer.use_default_system_prompt());
        if let Some(added_tokens) = tokenizer.added_tokens_decoder() {
            eprintln!("[DEBUG] Number of added tokens: {}", added_tokens.len());
        }
    }
    let model = models::from_pretrained(model_path)?;
    Ok((model, tokenizer))
}

/// Load model and tokenizer from an MLXPack file
pub fn load_from_mlxpack(pack_path: &Path, verbose: bool) -> Result<(Box<dyn LanguageModel>, Tokenizer), LoaderError> {
    mlxpack::load_from_mlxpack(pack_path, verbose)
        .map_err(Into::into)
}

pub fn use_gpu_if_available() {
    // Try to set GPU device, but fall back to CPU if Metal backend is not available
    let result = std::panic::catch_unwind(|| {
        mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    });

    match result {
        Ok(_) => {
            eprintln!("MLX: Using GPU device (Metal backend)");
        }
        Err(_) => {
            eprintln!("MLX: GPU backend not available, falling back to CPU");
            mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        }
    }
}