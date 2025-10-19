use crate::generate::generator::TokensGenerator;
use crate::models::LanguageModel;
use crate::tokenizer::{ApplyChatTemplateArgs, Conversation, Tokenizer, TokenizerError};
use mlx_rs::Array;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use tokenizers::{DecodeStream, Decoder, Normalizer, PostProcessor, PreTokenizer};

pub mod generator;
pub mod kv_cache;

#[derive(Debug, thiserror::Error)]
pub enum GenerationError {
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] TokenizerError),

    #[error("Model error: {0}")]
    ModelError(#[from] crate::models::ModelError),

    #[error("MLX exception: {0}")]
    MlxException(#[from] mlx_rs::error::Exception),

    #[error("Error: {0}")]
    Other(#[from] anyhow::Error),
}

pub struct GeneratorConfig {
    pub max_tokens: usize,
    pub temp: f32,
    pub top_p: f32,
    pub context_size: Option<usize>,
    pub num_sink_tokens: usize,
    pub eviction_batch_percent: f32,
    pub verbose: bool,
    pub skip_special_tokens: bool,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            max_tokens: 500,
            temp: 0.0,
            top_p: 1.0,
            context_size: None,
            num_sink_tokens: 4,
            eviction_batch_percent: 0.1,
            verbose: false,
            skip_special_tokens: true,
        }
    }
}

pub struct Generator<
    'a,
    'tok,
    M: LanguageModel,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
> {
    tokens_generator: TokensGenerator<'a, M>,
    decode_stream: DecodeStream<'tok, tokenizers::ModelWrapper, N, PT, PP, D>,
    eos_token_ids: Vec<u32>,
    max_tokens: usize,
    generated_tokens_count: usize,
}

impl<'a, 'tok, M: LanguageModel, N: Normalizer, PT: PreTokenizer, PP: PostProcessor, D: Decoder>
    Generator<'a, 'tok, M, N, PT, PP, D>
{
    fn new(
        tokens_generator: TokensGenerator<'a, M>,
        decode_stream: DecodeStream<'tok, tokenizers::ModelWrapper, N, PT, PP, D>,
        eos_token_ids: Vec<u32>,
        max_tokens: usize,
    ) -> Self {
        Self {
            tokens_generator,
            decode_stream,
            eos_token_ids,
            max_tokens,
            generated_tokens_count: 0,
        }
    }
}

impl<M: LanguageModel, N: Normalizer, PT: PreTokenizer, PP: PostProcessor, D: Decoder> Iterator
    for Generator<'_, '_, M, N, PT, PP, D>
{
    type Item = Result<String, GenerationError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've reached max_tokens
        if self.generated_tokens_count >= self.max_tokens {
            return None;
        }

        // Get next token from tokens_generator
        let token = match self.tokens_generator.next() {
            Some(Ok(token)) => token,
            Some(Err(e)) => return Some(Err(e.into())),
            None => {
                return None;
            }
        };

        let token_id = token.item::<u32>();
        if !self.eos_token_ids.is_empty() && self.eos_token_ids.contains(&token_id) {
            return None;
        }

        let chunk = match self.decode_stream.step(token_id) {
            Ok(chunk) => chunk,
            Err(e) => {
                return Some(Err(
                    anyhow::anyhow!("Failed to decode token stream: {}", e).into()
                ));
            }
        };

        self.generated_tokens_count += 1;

        match chunk {
            Some(s) => Some(Ok(s)),
            None => {
                // If decoder needs more tokens, recursively call next
                // This handles cases where decode_stream.step returns None
                self.next()
            }
        }
    }
}

pub fn generate<'a, M: LanguageModel>(
    model: &'a mut M,
    tokenizer: &'a Tokenizer,
    messages: &[Conversation],
    generator_config: GeneratorConfig,
) -> Result<impl Iterator<Item = Result<String, GenerationError>> + 'a, GenerationError> {

    let encoding = tokenizer.apply_chat_template_and_encode(
        ApplyChatTemplateArgs {
            conversation: messages.into(),
            add_generation_prompt: Some(true),
        },
        generator_config.verbose,
    )?;

    // Get context size from config or use model's max
    let max_context_size = generator_config.context_size.unwrap_or_else(|| model.max_context_size());

    let prompt_token_ids = encoding.get_ids();
    let prompt_token_count = prompt_token_ids.len();

    // Check if prompt exceeds context size at start - this is an error
    if prompt_token_count > max_context_size {
        return Err(GenerationError::Other(anyhow::anyhow!(
            "Input prompt ({} tokens) exceeds maximum context size ({} tokens). Please reduce prompt length.",
            prompt_token_count, max_context_size
        )));
    }

    if generator_config.verbose {
        eprintln!("[DEBUG] Prompt tokens: {}", prompt_token_count);
        eprintln!("[DEBUG] Max generation tokens: {}", generator_config.max_tokens);
        eprintln!("[DEBUG] Context size: {}", max_context_size);
        eprintln!("[DEBUG] Tokens available before context limit: {}", max_context_size.saturating_sub(prompt_token_count));
    }

    let prompt_tokens = Array::from(prompt_token_ids).index(NewAxis);
    let tokens_generator = TokensGenerator::new(
        model,
        prompt_tokens,
        generator_config.temp,
        generator_config.top_p,
        max_context_size,
        generator_config.num_sink_tokens,
        generator_config.eviction_batch_percent
    );
    let decoder = tokenizer.decode_stream(generator_config.skip_special_tokens);
    let eos_token_ids = tokenizer.eos_token_ids().map(|ids| ids.to_vec()).unwrap_or_default();
    Ok(Generator::new(
        tokens_generator,
        decoder,
        eos_token_ids,
        generator_config.max_tokens,
    ))
}
