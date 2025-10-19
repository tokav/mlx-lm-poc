use anyhow::Result;
use mlx_lm::generate::generate;
use mlx_lm::tokenizer::{Conversation, Role};
use clap::Parser;
use std::path::Path;

#[derive(Parser)]
#[command(about = "Mistral inference example")]
pub struct Cli {
    #[clap(long)]
    model_root: String,

    /// The message to be processed by the model
    #[clap(long, default_value = "In the begging the Universe was created.")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[clap(long, default_value = "500")]
    max_tokens: usize,

    /// The sampling temperature
    #[clap(long, default_value = "0.0")]
    temp: f32,

    /// The nucleus sampling threshold (top-p)
    #[clap(long, default_value = "1.0")]
    top_p: f32,

    /// Maximum context size (input + output tokens). If not specified, uses model's max_position_embeddings
    #[clap(long)]
    context_size: Option<usize>,

    /// Number of sink tokens (default: 4).
    #[clap(long, default_value = "4")]
    num_sink_tokens: usize,

    /// Cache eviction batch size as percentage (1-50, default: 10).
    /// When cache is full, evicts this percentage of tokens at once to create room.
    /// Higher values = fewer evictions but more tokens removed at once.
    #[clap(long, default_value = "10")]
    eviction_batch_percent: u8,

    /// The PRNG seed
    #[clap(long, default_value = "0")]
    seed: u64,

    /// Verbose mode
    #[clap(long, default_value_t = false)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Setting GPU as device...");
    mlx_lm::use_gpu_if_available();

    let cli = Cli::parse();
    mlx_rs::random::seed(cli.seed)?;

    println!("[INFO] Loading model... ");
    let model_root = Path::new(&cli.model_root);
    let (mut model, tokenizer) = if cli.model_root.ends_with(".mlxpack") {
        mlx_lm::load_from_mlxpack(model_root, cli.verbose)?
    } else {
        mlx_lm::load(model_root, cli.verbose)?
    };

    println!("[INFO] Formatting and encoding prompt... ");
    println!("Original prompt: {}", cli.prompt);

    println!("[INFO] Generating... ");
    let conversation = vec![Conversation {
        role: Role::User,
        content: cli.prompt,
    }];
    let generator = generate(
        &mut model,
        &tokenizer,
        &conversation,
        mlx_lm::generate::GeneratorConfig {
            max_tokens: cli.max_tokens,
            temp: cli.temp,
            top_p: cli.top_p,
            context_size: cli.context_size,
            num_sink_tokens: cli.num_sink_tokens,
            eviction_batch_percent: (cli.eviction_batch_percent as f32) / 100.0,
            verbose: cli.verbose,
            skip_special_tokens: true,
        },
    )?;

    // Iterate over the generator and print each decoded chunk
    for chunk in generator {
        let chunk = chunk?;
        print!("{}", chunk);
    }

    Ok(())
}
