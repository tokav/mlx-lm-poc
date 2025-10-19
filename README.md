# MLX-LM: Rust-based LLM Text Generation

> **Note**: This README and a significant portion of the codebase were generated with assistance from AI (Claude Code). This is an experimental proof-of-concept project.
> I'm effectively abandoning this project, It was meant to be a PoC mlx implementation for my iOS project but since it's inherently macOS only due to lack Metal support on iOS, it's not very useful for that purpose (mlx-swift does some magic with building mlx with iOS Matal framework support). Feel free to fork and continue development if interested.

A proof-of-concept implementation of Large Language Model (LLM) inference and text generation using [mlx-rs](https://github.com/oxideai/mlx-rs), the Rust bindings for Apple's MLX framework.

## Features

- **Supported Model Architectures**: Support for Mistral, Llama, Qwen3, and TinyLlama models
- **Quantization Support**: Load pre-quantized models (4-bit, 8-bit) for reduced memory footprint
- **Streaming Generation**: Real-time token-by-token text generation with configurable sampling
- **StreamingLLM Pattern**: Efficient KV-cache management with attention sink preservation for long contexts
- **MLXPack Format**: Pack and unpack models into single-file archives for easier distribution
- **Apple Silicon Optimized**: Leverages Metal GPU acceleration on macOS

## Supported Models

This implementation supports the following model architectures from HuggingFace:

### Mistral

- [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

### Llama

- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

### Qwen3

- [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

All models support both full-precision and quantized versions.

## Quantization

Models can be quantized using [mlx-lm](https://github.com/ml-explore/mlx-lm) to reduce memory usage and improve inference speed:

```bash
# Install mlx-lm
pip install mlx-lm

# Quantize a model to 4-bit with default group size (64)
mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.3 -q

# Custom quantization settings
mlx_lm.convert --hf-path meta-llama/Llama-3.2-1B-Instruct \
    --quantize \
    --q-bits 4 \
    --q-group-size 64 \
    --mlx-path ./models/llama-3.2-1b-4bit
```

The quantized models are automatically detected and loaded with the appropriate configuration.

## MLX-RS Patch

This project includes a patch for mlx-rs 0.25.2 that fixes parameter loading for quantized embeddings. The patch is located in `src/mlx_rs_patch.rs` and provides:

### QuantizedEmbedding Fix

**Issue**: In mlx-rs 0.25.2, `QuantizedEmbedding` doesn't expose its internal parameters (`scales`, `biases`, `inner`) because the `#[param]` attributes are missing. This prevents the model from loading quantized embedding weights from safetensors files.

**Solution**: The patch implements a custom `QuantizedEmbedding` struct with proper `#[param]` attributes:

```rust
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedEmbedding {
    pub group_size: i32,
    pub bits: i32,

    #[param]  // ← Fixed: Missing in mlx-rs 0.25.2
    pub scales: Param<Array>,

    #[param]  // ← Fixed: Missing in mlx-rs 0.25.2
    pub biases: Param<Array>,

    #[param]  // ← Fixed: Missing in mlx-rs 0.25.2
    pub inner: mlx_rs::nn::Embedding,
}
```

This ensures that `load_safetensors()` can properly populate the quantized embedding layer weights when loading pre-quantized models.

## Installation

### Prerequisites

- Rust 1.70 or later
- macOS with Apple Silicon (for Metal GPU acceleration)
- Git LFS (for downloading large model files)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/mlx-lm-poc.git
cd mlx-lm-poc

# Build the project
cargo build --release

# The binaries will be available at:
# - target/release/mlx_lm (main inference binary)
# - target/release/mlxpack (model packing utility)
```

## Usage

### Text Generation

The main binary `mlx_lm` performs LLM text generation with a conversational interface:

```bash
# Basic usage
cargo run --release -- \
    --model-root /path/to/model \
    --prompt "In the beginning the Universe was created."

# With custom generation parameters
cargo run --release -- \
    --model-root /path/to/mistral-7b-instruct \
    --prompt "Explain quantum computing" \
    --max-tokens 500 \
    --temp 0.7 \
    --top-p 0.9 \
    --context-size 4096
```

#### Command-line Parameters

| Parameter                  | Description                                               | Default                                      |
| -------------------------- | --------------------------------------------------------- | -------------------------------------------- |
| `--model-root`             | Path to model directory or .mlxpack file (required)       | -                                            |
| `--prompt`                 | Input text prompt                                         | "In the beginning the Universe was created." |
| `--max-tokens`             | Maximum number of tokens to generate                      | 500                                          |
| `--temp`                   | Sampling temperature (0.0 = greedy, higher = more random) | 0.0                                          |
| `--top-p`                  | Nucleus sampling threshold (0.0-1.0)                      | 1.0                                          |
| `--context-size`           | Maximum context size (input + output tokens)              | Model's max_position_embeddings              |
| `--num-sink-tokens`        | Number of attention sink tokens for StreamingLLM          | 4                                            |
| `--eviction-batch-percent` | Cache eviction batch size percentage (1-50)               | 10                                           |
| `--seed`                   | Random seed for reproducibility                           | 0                                            |
| `--verbose`                | Enable verbose logging                                    | false                                        |

#### Generation Parameters Explained

- **Temperature (`--temp`)**: Controls randomness in sampling

  - `0.0`: Greedy decoding (always picks most likely token)
  - `0.7`: Balanced creativity and coherence
  - `1.0+`: More creative but less coherent

- **Top-p (`--top-p`)**: Nucleus sampling threshold

  - `1.0`: Consider all tokens
  - `0.9`: Consider only top 90% probability mass (recommended)
  - Lower values make output more focused but less diverse

- **Context Size (`--context-size`)**: Maximum total tokens (prompt + generation)

  - When exceeded, uses StreamingLLM pattern to evict middle tokens
  - Preserves attention sinks and recent tokens for stable generation

- **Eviction Batch Percent (`--eviction-batch-percent`)**: Controls cache eviction strategy
  - When cache fills, evicts this percentage of tokens at once
  - `10` (default): Evicts 10% of cache, reducing eviction frequency
  - Higher values = fewer evictions but more tokens removed at once

### MLXPack Utility (Implementing proper GGUF support would take too much time for PoC implementation)

The `mlxpack` utility allows you to pack HuggingFace model directories into single-file archives for easier distribution and faster loading:

#### Pack a Model

```bash
# Pack a model directory into a single file
cargo run --release --bin mlxpack -- pack \
    --input /path/to/model/directory \
    --output model.mlxpack

# Example output:
# [INFO] Packing model from: "/path/to/mistral-7b-4bit"
# [INFO] Model architecture: MistralForCausalLM
# [INFO] Safetensors files to pack:
#   - model.safetensors (13421772800 bytes)
# [INFO] ✓ Packed successfully!
# [INFO] Output size: 13422109876 bytes (12.50 GB)
```

#### View MLXPack Information

```bash
# Display metadata about a packed model
cargo run --release --bin mlxpack -- info \
    --input model.mlxpack

# Example output:
# === MLXPack File Information ===
#
# File size: 13422109876 bytes (12.50 GB)
#
# --- Model Configuration ---
# Architecture: MistralForCausalLM
# Hidden size: 4096
# Num layers: 32
# Num heads: 32
# Vocab size: 32000
#
# Quantization:
#   Bits: 4
#   Group size: 64
#   Mode: channel_wise
#
# --- Tokenizer Configuration ---
# BOS token ID: 1
# EOS token ID: 2
# Has chat template: true
#
# --- Safetensors Files ---
# Total files: 1
#   model.safetensors (13421772800 bytes, 12800.00 MB)
# Total weights size: 13421772800 bytes (12.50 GB)
```

#### Extract Files from MLXPack

```bash
# Extract all safetensors files
cargo run --release --bin mlxpack -- extract \
    --input model.mlxpack \
    --output /path/to/output/directory

# Extract a specific file
cargo run --release --bin mlxpack -- extract \
    --input model.mlxpack \
    --output /path/to/output \
    --file model.safetensors
```

#### Use MLXPack with Text Generation

```bash
# Generate text directly from an MLXPack file
cargo run --release -- \
    --model-root model.mlxpack \
    --prompt "Write a short story about AI"
```

## Project Structure

```
mlx-lm-poc/
├── src/
│   ├── bin/
│   │   └── mlxpack.rs          # MLXPack CLI utility
│   ├── models/
│   │   ├── mod.rs              # Model trait and loading logic
│   │   ├── mistral.rs          # Mistral model implementation
│   │   ├── llama.rs            # Llama model implementation
│   │   └── qwen3.rs            # Qwen3 model implementation
│   ├── generate/
│   │   ├── generator.rs        # Token generation and sampling
│   │   ├── kv_cache.rs         # StreamingLLM KV-cache management
│   │   └── mod.rs              # Generation interface
│   ├── mlxpack/
│   │   ├── writer.rs           # MLXPack file writer
│   │   ├── reader.rs           # MLXPack file reader
│   │   ├── loader.rs           # Load models from MLXPack
│   │   └── mod.rs              # MLXPack public API
│   ├── mlx_rs_patch.rs         # Fixes for mlx-rs 0.25.2
│   ├── quantized.rs            # Quantized model loading
│   ├── tokenizer.rs            # HuggingFace tokenizer wrapper
│   ├── lib.rs                  # Library interface
│   └── main.rs                 # Main CLI application
├── Cargo.toml                  # Rust dependencies
└── README.md
```

## Technical Details

### StreamingLLM Implementation

This project implements the [StreamingLLM](https://arxiv.org/abs/2309.17453) pattern for efficient long-context generation:

1. **Attention Sinks**: Preserves the first N tokens (default: 4) which models use as attention anchors
2. **Sliding Window**: Maintains recent tokens for context
3. **Batch Eviction**: Removes middle tokens in batches when cache fills, reducing eviction frequency

This prevents model degeneration during long generations while maintaining memory efficiency.

### Key-Value Cache Structure

Each transformer layer maintains a separate KV-cache with:

- **Keys**: `[batch, n_heads, seq_len, head_dim]`
- **Values**: `[batch, n_heads, seq_len, head_dim]`
- **Offset Tracking**: Maintains absolute position for RoPE calculations

Caches are managed synchronously across all layers to ensure consistent sequence lengths.

### Sampling Strategies

- **Greedy Decoding** (`temp=0.0`): Always selects the most likely token
- **Temperature Sampling** (`temp>0.0`): Scales logits to control randomness
- **Nucleus Sampling** (`top_p<1.0`): Samples from tokens comprising top-p probability mass

## Performance

Performance varies by model size and quantization:

| Model          | Precision | Memory  | Tokens/sec\* |
| -------------- | --------- | ------- | ------------ |
| Mistral-7B     | 4-bit     | ~4.5 GB | ~40-50       |
| Llama-3.2-1B   | 4-bit     | ~1.2 GB | ~120-150     |
| TinyLlama-1.1B | 4-bit     | ~1.0 GB | ~130-160     |
| Qwen3-4B       | 4-bit     | ~2.8 GB | ~60-80       |

\*Approximate values on M1 Pro/Max, actual performance varies by prompt length and generation settings

## License

This project is provided as-is for educational and research purposes. Please refer to the individual model licenses on HuggingFace for model usage terms.

## Contributing

This is a proof-of-concept project. Contributions, bug reports, and suggestions are welcome via GitHub issues and pull requests.

## Acknowledgments

- [mlx-rs](https://github.com/oxideai/mlx-rs) - Rust bindings for Apple's MLX framework
- [Apple MLX](https://github.com/ml-explore/mlx) - Array framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - Python LLM tools for MLX
- [HuggingFace](https://huggingface.co) - Model hosting and tokenizers library

## References

- [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [MLX: An array framework for Apple silicon](https://ml-explore.github.io/mlx/)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
