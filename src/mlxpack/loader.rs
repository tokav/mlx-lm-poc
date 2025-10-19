/// MLXPack loader - loads models and tokenizers from MLXPack files

use super::{MLXPackReader, MLXPackError, Result};
use crate::models::{LanguageModel, ModelArgs};
use crate::tokenizer::Tokenizer;
use std::path::Path;
use std::collections::HashMap;
use mlx_rs::Array;
use minijinja::Environment;

/// Load model and tokenizer from an MLXPack file
pub fn load_from_mlxpack(pack_path: &Path, verbose: bool) -> Result<(Box<dyn LanguageModel>, Tokenizer)> {
    if verbose {
        eprintln!("[INFO] Loading from MLXPack file: {:?}", pack_path);
    }

    let mut reader = MLXPackReader::open(pack_path)?;

    // Build tokenizer from embedded data
    let tokenizer = build_tokenizer_from_metadata(&reader, verbose)?;

    // Load model
    let model = load_model_from_mlxpack(&mut reader, verbose)?;

    Ok((model, tokenizer))
}

/// Build tokenizer from MLXPack metadata
fn build_tokenizer_from_metadata(
    reader: &MLXPackReader,
    verbose: bool,
) -> Result<Tokenizer> {
    let metadata = reader.metadata();

    // Parse tokenizer.json
    let inner_tokenizer = tokenizers::Tokenizer::from_bytes(metadata.tokenizer_json.as_bytes())
        .map_err(|e| MLXPackError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to parse tokenizer.json: {}", e)
        )))?;

    // Setup Jinja environment for chat template
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // Get chat template from metadata or tokenizer_config
    let chat_template = metadata.chat_template.clone()
        .or_else(|| {
            metadata.tokenizer_config.as_ref()
                .and_then(|cfg| cfg.chat_template.clone())
        });

    if let Some(ref template) = chat_template {
        env.add_template_owned("default".to_string(), template.clone())
            .map_err(|e| MLXPackError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to add chat template: {}", e)
            )))?;
    }

    // Construct Tokenizer
    let tokenizer = Tokenizer::from_components(
        inner_tokenizer,
        env,
        metadata.tokenizer_config.clone(),
        metadata.generation_config.clone(),
    );

    if verbose {
        print_tokenizer_debug_info(&tokenizer);
    }

    Ok(tokenizer)
}

/// Load model from MLXPack reader
fn load_model_from_mlxpack(
    reader: &mut MLXPackReader,
    verbose: bool,
) -> Result<Box<dyn LanguageModel>> {
    let model_args = reader.config().clone();

    // Determine architecture from config
    let architecture = detect_architecture(&model_args)?;

    if verbose {
        eprintln!("[INFO] Model architecture: {}", architecture);
        eprintln!("[INFO] Safetensors files in pack: {:?}", reader.safetensors_filenames());
    }

    // Create model structure and load weights
    let model: Box<dyn LanguageModel> = match architecture.as_str() {
        "MistralForCausalLM" => {
            let mut model = crate::models::mistral::Model::new(&model_args)
                .map_err(|e| MLXPackError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create Mistral model: {}", e)
                )))?;

            if let Some(quant_config) = &model_args.quantization {
                if verbose {
                    eprintln!(
                        "[INFO] Pre-quantized model detected ({}bit, group_size={})",
                        quant_config.bits, quant_config.group_size
                    );
                    eprintln!("[INFO] Quantizing model structure before loading weights");
                }
                model = mlx_rs::nn::quantize(
                    model,
                    Some(quant_config.group_size),
                    Some(quant_config.bits),
                ).map_err(|e| MLXPackError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to quantize model: {}", e)
                )))?;
            }
            load_weights_from_mlxpack(&mut model, reader, &model_args, verbose)?;
            Box::new(model)
        }
        "Qwen3ForCausalLM" => {
            let mut model = crate::models::qwen3::Model::new(&model_args)
                .map_err(|e| MLXPackError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create Qwen3 model: {}", e)
                )))?;

            if let Some(quant_config) = &model_args.quantization {
                if verbose {
                    eprintln!(
                        "[INFO] Pre-quantized model detected ({}bit, group_size={})",
                        quant_config.bits, quant_config.group_size
                    );
                    eprintln!("[INFO] Quantizing model structure before loading weights");
                }
                model = mlx_rs::nn::quantize(
                    model,
                    Some(quant_config.group_size),
                    Some(quant_config.bits),
                ).map_err(|e| MLXPackError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to quantize model: {}", e)
                )))?;
            }
            load_weights_from_mlxpack(&mut model, reader, &model_args, verbose)?;
            Box::new(model)
        }
        "LlamaForCausalLM" => {
            let mut model = crate::models::llama::Model::new(&model_args)
                .map_err(|e| MLXPackError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create Llama model: {}", e)
                )))?;

            if let Some(quant_config) = &model_args.quantization {
                if verbose {
                    eprintln!(
                        "[INFO] Pre-quantized model detected ({}bit, group_size={})",
                        quant_config.bits, quant_config.group_size
                    );
                    eprintln!("[INFO] Quantizing model structure before loading weights");
                }
                model = mlx_rs::nn::quantize(
                    model,
                    Some(quant_config.group_size),
                    Some(quant_config.bits),
                ).map_err(|e| MLXPackError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to quantize model: {}", e)
                )))?;
            }
            load_weights_from_mlxpack(&mut model, reader, &model_args, verbose)?;
            Box::new(model)
        }
        arch => {
            return Err(MLXPackError::InvalidFormat(format!(
                "Unsupported model architecture: {}",
                arch
            )))
        }
    };

    Ok(model)
}

/// Detect model architecture from config
fn detect_architecture(model_args: &ModelArgs) -> Result<String> {
    let architecture = model_args
        .architectures
        .as_ref()
        .and_then(|archs| archs.first())
        .ok_or_else(|| {
            MLXPackError::InvalidFormat("No architecture specified in config".to_string())
        })?;

    Ok(architecture.clone())
}

/// Load weights from MLXPack into model
fn load_weights_from_mlxpack<T: mlx_rs::module::ModuleParametersExt>(
    model: &mut T,
    reader: &mut MLXPackReader,
    model_args: &ModelArgs,
    verbose: bool,
) -> Result<()> {
    // Clone filenames to avoid borrow checker issues
    let filenames: Vec<String> = reader.safetensors_filenames()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    for filename in filenames {
        if verbose {
            eprintln!("[INFO] Loading safetensors: {}", filename);
        }
        let arrays = reader.load_safetensors_arrays(&filename)?;

        // Apply the same loading logic as models/mod.rs
        if model_args.quantization.is_some() {
            // Quantized loading with parameter name remapping
            load_quantized_arrays(model, arrays)?;
        } else {
            // Non-quantized loading
            load_arrays(model, arrays)?;
        }
    }

    // Loading is lazy, eval after loading
    model.eval()
        .map_err(|e| MLXPackError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to eval model: {}", e)
        )))?;

    Ok(())
}

/// Load non-quantized arrays into model
fn load_arrays<T: mlx_rs::module::ModuleParametersExt>(
    model: &mut T,
    arrays: HashMap<String, Array>,
) -> Result<()> {
    let mut params = model.parameters_mut().flatten();
    let mut loaded_count = 0;
    let mut missing_count = 0;

    for (key, value) in arrays {
        if let Some(param) = params.get_mut(&*key) {
            **param = value;
            loaded_count += 1;
        } else {
            eprintln!("[WARNING] Parameter {} not found in model", key);
            missing_count += 1;
        }
    }

    if missing_count > 0 {
        eprintln!(
            "[WARNING] {} parameters from safetensors were not found in model",
            missing_count
        );
    }
    eprintln!("[INFO] Loaded {} parameters", loaded_count);
    Ok(())
}

/// Load quantized arrays with parameter name remapping
/// Same logic as quantized::load_quantized_safetensors
fn load_quantized_arrays<T: mlx_rs::module::ModuleParametersExt>(
    model: &mut T,
    arrays: HashMap<String, Array>,
) -> Result<()> {
    // Remap parameter names
    let mut remapped_tensors: HashMap<String, Array> = HashMap::new();

    for (name, array) in arrays {
        // Remap quantized layer parameters (self_attn, mlp, and embed_tokens)
        if name.contains(".embed_tokens.")
            || name.contains("lm_head.")
            || (name.contains(".self_attn.")
                && !name.contains(".q_norm.")
                && !name.contains(".k_norm.")
                && !name.contains(".rope."))
            || name.contains(".mlp.")
        {
            let mapped_name = if name.ends_with(".weight") && !name.contains(".inner.") {
                // For weight parameters, insert ".inner" before ".weight"
                let last_dot = name.rfind('.').unwrap();
                format!("{}.inner.{}", &name[..last_dot], &name[last_dot + 1..])
            } else {
                // For scales and biases, keep as-is
                name.to_string()
            };
            remapped_tensors.insert(mapped_name, array);
        } else {
            // Non-quantized layers: keep as-is
            remapped_tensors.insert(name.to_string(), array);
        }
    }

    // Update model parameters with remapped names
    let mut params = model.parameters_mut().flatten();
    let mut loaded_count = 0;
    let mut missing_count = 0;

    for (key, value) in remapped_tensors {
        if let Some(param) = params.get_mut(&*key) {
            **param = value;
            loaded_count += 1;
        } else {
            eprintln!("[WARNING] Parameter {} not found in model", key);
            missing_count += 1;
        }
    }

    if missing_count > 0 {
        eprintln!(
            "[WARNING] {} parameters from safetensors were not found in model",
            missing_count
        );
    }
    eprintln!("[INFO] Loaded {} parameters", loaded_count);

    Ok(())
}

/// Print tokenizer debug information
fn print_tokenizer_debug_info(tokenizer: &Tokenizer) {
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
