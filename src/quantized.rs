use std::collections::HashMap;
use std::path::Path;

use crate::models::ModelError;

// Load quantized model parameters from a safetensors file into the given model.
pub fn load_quantized_safetensors<T: mlx_rs::module::ModuleParametersExt>(
    model: &mut T,
    path: &Path,
) -> Result<(), ModelError> {
    use mlx_rs::Array;

    // Load all arrays from safetensors
    let loaded = Array::load_safetensors(path)?;

    // Remap parameter names
    let mut remapped_tensors: HashMap<String, Array> = HashMap::new();

    for (name, array) in loaded {
        // Remap quantized layer parameters (self_attn, mlp, and now embed_tokens)
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
                let new_name = format!("{}.inner.{}", &name[..last_dot], &name[last_dot + 1..]);
                new_name
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
            "[WARNING] {} parameters from safetensors file were not found in model",
            missing_count
        );
    }
    eprintln!("[INFO] Loaded {} parameters", loaded_count);

    // Loading is lazy, eval after loading
    model.eval()?;

    Ok(())
}
