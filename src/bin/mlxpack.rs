/// CLI tool for packing and unpacking MLXPack files

use anyhow::Result;
use mlx_lm::mlxpack::{MLXPackWriter, MLXPackReader};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mlxpack")]
#[command(about = "Pack and unpack MLX models into single files", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pack a HuggingFace model directory into a single MLXPack file
    Pack {
        /// Path to HuggingFace model directory
        #[arg(short, long)]
        input: PathBuf,

        /// Output MLXPack file path
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Show information about an MLXPack file
    Info {
        /// Path to MLXPack file
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Extract safetensors files from an MLXPack file
    Extract {
        /// Path to MLXPack file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for extracted files
        #[arg(short, long)]
        output: PathBuf,

        /// Specific safetensors file to extract (optional, extracts all if not specified)
        #[arg(short, long)]
        file: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pack { input, output } => {
            pack_model(&input, &output)?;
        }
        Commands::Info { input } => {
            show_info(&input)?;
        }
        Commands::Extract { input, output, file } => {
            extract_files(&input, &output, file.as_deref())?;
        }
    }

    Ok(())
}

fn pack_model(input: &PathBuf, output: &PathBuf) -> Result<()> {
    println!("[INFO] Packing model from: {:?}", input);
    println!("[INFO] Output file: {:?}", output);

    let mut writer = MLXPackWriter::new(input)?;

    // Show what will be packed
    let metadata = writer.metadata();
    println!("[INFO] Model architecture: {}",
        serde_json::to_value(&metadata.config)?
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
    );
    println!("[INFO] Safetensors files to pack:");
    for entry in &metadata.safetensors_files {
        println!("  - {} ({} bytes)", entry.name, entry.size);
    }

    // Pack
    writer.pack(output)?;

    // Show final size
    let packed_size = std::fs::metadata(output)?.len();
    println!("[INFO] ✓ Packed successfully!");
    println!("[INFO] Output size: {} bytes ({:.2} GB)",
        packed_size,
        packed_size as f64 / 1_073_741_824.0
    );

    Ok(())
}

fn show_info(input: &PathBuf) -> Result<()> {
    println!("[INFO] Reading MLXPack file: {:?}", input);

    let reader = MLXPackReader::open(input)?;
    let metadata = reader.metadata();

    // File size
    let file_size = std::fs::metadata(input)?.len();
    println!("\n=== MLXPack File Information ===\n");
    println!("File size: {} bytes ({:.2} GB)", file_size, file_size as f64 / 1_073_741_824.0);

    // Model config
    println!("\n--- Model Configuration ---");
    println!("Architecture: {}",
        metadata.config.architectures
            .as_ref()
            .and_then(|archs| archs.first())
            .map(|s| s.as_str())
            .unwrap_or("unknown")
    );
    println!("Hidden size: {}", metadata.config.hidden_size);
    println!("Num layers: {}", metadata.config.num_hidden_layers);
    println!("Num heads: {}", metadata.config.num_attention_heads);
    println!("Vocab size: {}", metadata.config.vocab_size);

    if let Some(quant) = &metadata.config.quantization {
        println!("\nQuantization:");
        println!("  Bits: {}", quant.bits);
        println!("  Group size: {}", quant.group_size);
        if let Some(mode) = &quant.mode {
            println!("  Mode: {}", mode);
        }
    }

    // Tokenizer info
    println!("\n--- Tokenizer Configuration ---");
    if let Some(gen_config) = &metadata.generation_config {
        if let Some(bos) = &gen_config.bos_token_id {
            println!("BOS token ID: {:?}", bos);
        }
        if let Some(eos) = &gen_config.eos_token_id {
            println!("EOS token ID: {:?}", eos);
        }
    }
    println!("Has chat template: {}", metadata.chat_template.is_some());

    // Safetensors files
    println!("\n--- Safetensors Files ---");
    println!("Total files: {}", metadata.safetensors_files.len());
    let mut total_size = 0u64;
    for entry in &metadata.safetensors_files {
        println!("  {} ({} bytes, {:.2} MB)",
            entry.name,
            entry.size,
            entry.size as f64 / 1_048_576.0
        );
        total_size += entry.size;
    }
    println!("Total weights size: {} bytes ({:.2} GB)", total_size, total_size as f64 / 1_073_741_824.0);

    Ok(())
}

fn extract_files(input: &PathBuf, output: &PathBuf, specific_file: Option<&str>) -> Result<()> {
    println!("[INFO] Extracting from: {:?}", input);
    println!("[INFO] Output directory: {:?}", output);

    // Create output directory
    std::fs::create_dir_all(output)?;

    let mut reader = MLXPackReader::open(input)?;

    // Clone filenames to avoid borrow checker issues
    let filenames: Vec<String> = reader.safetensors_filenames()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    match specific_file {
        Some(file) => {
            // Extract specific file
            if !filenames.iter().any(|f| f == file) {
                eprintln!("[ERROR] File '{}' not found in pack", file);
                eprintln!("Available files: {:?}", filenames);
                return Err(anyhow::anyhow!("File not found"));
            }

            let output_path = output.join(file);
            println!("[INFO] Extracting: {}", file);
            reader.extract_safetensors_to_file(file, &output_path)?;
            println!("[INFO] ✓ Extracted to: {:?}", output_path);
        }
        None => {
            // Extract all files
            println!("[INFO] Extracting {} files...", filenames.len());
            for filename in &filenames {
                let output_path = output.join(filename);
                println!("[INFO] Extracting: {}", filename);
                reader.extract_safetensors_to_file(filename, &output_path)?;
            }
            println!("[INFO] ✓ All files extracted successfully!");
        }
    }

    Ok(())
}
