use std::{
    collections::HashMap,
    fs::read_to_string,
    ops::{Deref, DerefMut},
    path::Path,
};

use minijinja::{context, Environment};
use serde::{Deserialize, Deserializer, Serialize};
use tokenizers::Encoding;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),

    #[error("Jinja rendering error: {0}")]
    JinjaError(#[from] minijinja::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Final message not found in rendered chat")]
    FinalMsgNotInChat,

    #[error("Chat template not found in tokenizer config")]
    NoChatTemplate,
}

// ============================================================================
// Special Tokens Configuration
// ============================================================================

/// Custom deserializer that handles both single u32 and Vec<u32> for token IDs
fn deserialize_token_ids<'de, D>(deserializer: D) -> Result<Option<Vec<u32>>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TokenIds {
        Single(u32),
        Multiple(Vec<u32>),
    }

    let value = Option::<TokenIds>::deserialize(deserializer)?;
    Ok(value.map(|v| match v {
        TokenIds::Single(id) => vec![id],
        TokenIds::Multiple(ids) => ids,
    }))
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct GenerationConfig {
    #[serde(default, deserialize_with = "deserialize_token_ids")]
    pub bos_token_id: Option<Vec<u32>>,
    #[serde(default, deserialize_with = "deserialize_token_ids")]
    pub eos_token_id: Option<Vec<u32>>,
    #[serde(default, deserialize_with = "deserialize_token_ids")]
    pub pad_token_id: Option<Vec<u32>>,
}

/// Configuration for an added token in the tokenizer
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct AddedTokenDecoder {
    pub content: String,
    #[serde(default)]
    pub lstrip: bool,
    #[serde(default)]
    pub normalized: bool,
    #[serde(default)]
    pub rstrip: bool,
    #[serde(default)]
    pub single_word: bool,
    #[serde(default)]
    pub special: bool,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub bos_token: Option<String>,
    #[serde(default)]
    pub eos_token: Option<String>,
    #[serde(default)]
    pub pad_token: Option<String>,
    #[serde(default)]
    pub add_bos_token: Option<bool>,
    #[serde(default)]
    pub add_eos_token: Option<bool>,
    #[serde(default)]
    pub add_prefix_space: Option<bool>,
    #[serde(default)]
    pub added_tokens_decoder: Option<HashMap<String, AddedTokenDecoder>>,
    #[serde(default)]
    pub clean_up_tokenization_spaces: Option<bool>,
    #[serde(default)]
    pub spaces_between_special_tokens: Option<bool>,
    #[serde(default)]
    pub use_default_system_prompt: Option<bool>,
}

// ============================================================================
// Tokenizer Wrapper
// ============================================================================

/// Wrapper around [`tokenizers::Tokenizer`] with chat template support
/// and special token handling
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    env: Environment<'static>,
    tokenizer_config: Option<TokenizerConfig>,
    generation_config: Option<GenerationConfig>,
}

impl Tokenizer {
    /// Create a tokenizer from components (for internal use)
    pub(crate) fn from_components(
        inner: tokenizers::Tokenizer,
        env: Environment<'static>,
        tokenizer_config: Option<TokenizerConfig>,
        generation_config: Option<GenerationConfig>,
    ) -> Self {
        Self {
            inner,
            env,
            tokenizer_config,
            generation_config,
        }
    }

    /// Load tokenizer from model root directory
    ///
    /// Loads:
    /// - tokenizer.json (required)
    /// - tokenizer_config.json (optional, for chat template and special tokens)
    /// - generation_config.json (optional, for token IDs)
    pub fn from_model_dir(model_root: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let model_root = model_root.as_ref();

        // Load tokenizer
        let tokenizer_path = model_root.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(tokenizer_path)?;
        // Load tokenizer config (optional)
        let tokenizer_config_path = model_root.join("tokenizer_config.json");
        let tokenizer_config = if tokenizer_config_path.exists() {
            let content = read_to_string(&tokenizer_config_path)?;
            Some(serde_json::from_str::<TokenizerConfig>(&content)?)
        } else {
            None
        };

        // Load generation config (optional)
        let generation_config_path = model_root.join("generation_config.json");
        let generation_config = if generation_config_path.exists() {
            let content = read_to_string(&generation_config_path)?;
            Some(serde_json::from_str::<GenerationConfig>(&content)?)
        } else {
            None
        };

        // Extract chat template from either chat_template.jinja file or tokenizer_config.json
        let chat_template_file = model_root.join("chat_template.jinja");
        let chat_template = if chat_template_file.exists() {
            // MLX-LM quantized models store chat template in separate file
            Some(read_to_string(&chat_template_file)?)
        } else {
            // Standard models store chat template in tokenizer_config.json
            tokenizer_config
                .as_ref()
                .and_then(|cfg| cfg.chat_template.clone())
        };

        // Initialize Jinja environment
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        match chat_template {
            Some(ref template) => {
                env.add_template_owned("default".to_string(), template.clone())?;
            }
            None => {
                // No chat template, but we can still use the environment for future templates
            }
        }

        Ok(Self {
            inner,
            env,
            tokenizer_config,
            generation_config,
        })
    }

    /// Get BOS token ID (returns first if multiple)
    pub fn bos_token_id(&self) -> Option<u32> {
        self.generation_config
            .as_ref()
            .and_then(|cfg| cfg.bos_token_id.as_ref())
            .and_then(|ids| ids.first().copied())
    }

    /// Get all BOS token IDs
    pub fn bos_token_ids(&self) -> Option<&[u32]> {
        self.generation_config
            .as_ref()
            .and_then(|cfg| cfg.bos_token_id.as_ref())
            .map(|ids| ids.as_slice())
    }

    /// Get EOS token ID (returns first if multiple)
    pub fn eos_token_id(&self) -> Option<u32> {
        self.generation_config
            .as_ref()
            .and_then(|cfg| cfg.eos_token_id.as_ref())
            .and_then(|ids| ids.first().copied())
    }

    /// Get all EOS token IDs
    pub fn eos_token_ids(&self) -> Option<&[u32]> {
        self.generation_config
            .as_ref()
            .and_then(|cfg| cfg.eos_token_id.as_ref())
            .map(|ids| ids.as_slice())
    }

    /// Get PAD token ID (returns first if multiple)
    pub fn pad_token_id(&self) -> Option<u32> {
        self.generation_config
            .as_ref()
            .and_then(|cfg| cfg.pad_token_id.as_ref())
            .and_then(|ids| ids.first().copied())
    }

    /// Get all PAD token IDs
    pub fn pad_token_ids(&self) -> Option<&[u32]> {
        self.generation_config
            .as_ref()
            .and_then(|cfg| cfg.pad_token_id.as_ref())
            .map(|ids| ids.as_slice())
    }

    /// Get BOS token string
    pub fn bos_token(&self) -> Option<&str> {
        self.tokenizer_config
            .as_ref()
            .and_then(|cfg| cfg.bos_token.as_deref())
    }

    /// Get EOS token string
    pub fn eos_token(&self) -> Option<&str> {
        self.tokenizer_config
            .as_ref()
            .and_then(|cfg| cfg.eos_token.as_deref())
    }

    /// Get PAD token string
    pub fn pad_token(&self) -> Option<&str> {
        self.tokenizer_config
            .as_ref()
            .and_then(|cfg| cfg.pad_token.as_deref())
    }

    /// Check if BOS token should be added automatically
    pub fn add_bos_token(&self) -> bool {
        self.tokenizer_config
            .as_ref()
            .and_then(|cfg| cfg.add_bos_token)
            .unwrap_or(false)
    }

    /// Check if EOS token should be added automatically
    pub fn add_eos_token(&self) -> bool {
        self.tokenizer_config
            .as_ref()
            .and_then(|cfg| cfg.add_eos_token)
            .unwrap_or(false)
    }

    /// Check if prefix space should be added
    pub fn add_prefix_space(&self) -> bool {
        self.tokenizer_config
            .as_ref()
            .and_then(|cfg| cfg.add_prefix_space)
            .unwrap_or(false)
    }

    /// Get added tokens decoder map
    pub fn added_tokens_decoder(&self) -> Option<&HashMap<String, AddedTokenDecoder>> {
        self.tokenizer_config
            .as_ref()
            .and_then(|cfg| cfg.added_tokens_decoder.as_ref())
    }

    /// Check if tokenization spaces should be cleaned up
    pub fn clean_up_tokenization_spaces(&self) -> bool {
        self.tokenizer_config
            .as_ref()
            .and_then(|cfg| cfg.clean_up_tokenization_spaces)
            .unwrap_or(true) // Default is true for most tokenizers
    }

    /// Check if spaces should be added between special tokens
    pub fn spaces_between_special_tokens(&self) -> bool {
        self.tokenizer_config
            .as_ref()
            .and_then(|cfg| cfg.spaces_between_special_tokens)
            .unwrap_or(true) // Default is true for most tokenizers
    }

    /// Check if default system prompt should be used
    pub fn use_default_system_prompt(&self) -> bool {
        self.tokenizer_config
            .as_ref()
            .and_then(|cfg| cfg.use_default_system_prompt)
            .unwrap_or(false)
    }

    /// Get chat template
    pub fn chat_template(&self) -> Option<String> {
        match self.env.get_template("default") {
            Ok(template) => Some(template.source().to_string()),
            Err(_) => None,
        }
    }

    pub fn apply_chat_template(
        &self,
        args: ApplyChatTemplateArgs,
    ) -> Result<String, TokenizerError>
    {
        let bos_token = self.bos_token().unwrap_or("").to_string();
        let eos_token = self.eos_token().unwrap_or("").to_string();
        match self.env.get_template("default") {
            Ok(template) => {
                let ApplyChatTemplateArgs {
                    conversation: conversations,
                    add_generation_prompt,
                } = args;

                let add_generation_prompt = add_generation_prompt.unwrap_or(false);

                let rendered_chat = template.render(context! {
                        messages => conversations,
                        bos_token => bos_token,
                        eos_token => eos_token,
                        add_generation_prompt => add_generation_prompt,
                    })?;


                Ok(rendered_chat)
            }
            Err(_) => Err(TokenizerError::NoChatTemplate),
        }
    }

    pub fn apply_chat_template_and_encode(
        &self,
        args: ApplyChatTemplateArgs,
        verbose: bool,
    ) -> Result<Encoding, TokenizerError>
    {
        let chat = self.apply_chat_template(args)?;
        if verbose {
            eprintln!("[DEBUG] --- Rendered Chat ---");
            eprintln!("[DEBUG] {}", chat);
            eprintln!("[DEBUG] -----------------------");
        }
        self.inner.encode(chat, false).map_err(Into::into)
    }

    pub fn format_user_message(
        &self,
        message: &str,
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        match self.chat_template() {
            None => Ok(message.to_string()),
            Some(_) => {
                let conversation = vec![Conversation {
                    role: Role::User,
                    content: message.to_string(),
                }];

                let args = ApplyChatTemplateArgs {
                    conversation,
                    add_generation_prompt: Some(add_generation_prompt),
                };
                let rendered = self.apply_chat_template(args)?;
                Ok(rendered)
            }
        }
    }

    pub fn encode_user_message(
        &self,
        message: &str,
        add_generation_prompt: bool,
        verbose: bool,
    ) -> Result<Encoding, TokenizerError> {
        match self.chat_template() {
            None => self.inner.encode(message, false).map_err(Into::into),
            Some(_) => {
                let conversation = vec![Conversation {
                    role: Role::User,
                    content: message.to_string(),
                }];

                let args = ApplyChatTemplateArgs {
                    conversation,
                    add_generation_prompt: Some(add_generation_prompt),
                };

                let encodings = self.apply_chat_template_and_encode(args, verbose)?;
                Ok(encodings)
            }
        }
    }
}

impl Deref for Tokenizer {
    type Target = tokenizers::Tokenizer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Tokenizer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize)]
pub struct Conversation {
    pub role: Role,
    pub content: String,
}

#[derive(Default)]
pub struct ApplyChatTemplateArgs
{
    pub conversation: Vec<Conversation>,
    pub add_generation_prompt: Option<bool>,
}

