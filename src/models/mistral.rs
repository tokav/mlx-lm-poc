use crate::models::{LanguageModel, ModelArgs, ModelError, ModelInput, ModelOutput};
use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    fast::{ScaledDotProductAttentionMask, scaled_dot_product_attention},
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    quantization::MaybeQuantized,
};
use crate::mlx_rs_patch::Embedding;
// ============================================================================
// SelfAttn (Self-Attention Layer)
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct SelfAttn {
    n_heads: i32,
    n_kv_heads: i32,
    repeats: i32,
    scale: f32,

    #[quantizable]
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,

    #[param]
    rope: nn::Rope,
}

impl SelfAttn {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let n_heads = args.n_heads();
        let n_kv_heads = args.n_kv_heads();
        let head_dim = args.head_dim();
        let dim = args.dim();
        let repeats = n_heads / n_kv_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;
        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(false)
            .base(
                args.rope_theta
                    .unwrap_or(crate::models::ModelArgs::DEFAULT_ROPE_THETA),
            )
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            repeats,
            scale,
            q_proj: MaybeQuantized::new(q_proj),
            k_proj: MaybeQuantized::new(k_proj),
            v_proj: MaybeQuantized::new(v_proj),
            o_proj: MaybeQuantized::new(o_proj),
            rope,
        })
    }
}

struct AttentionInput<'a> {
    x: &'a Array,
    mask: Option<ScaledDotProductAttentionMask<'a>>,
    cache: &'a mut crate::generate::kv_cache::KeyValueCache,
}

struct AttentionOutput {
    output: Array,
}

impl Module<AttentionInput<'_>> for SelfAttn {
    type Output = AttentionOutput;

    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        // NOTE: this will panic if the input shape is not correct
        let B = x.shape()[0];
        let L = x.shape()[1];

        let mut queries = self.q_proj.forward(x)?;
        let mut keys = self.k_proj.forward(x)?;
        let mut values = self.v_proj.forward(x)?;

        // Prepare the queries, keys, and values for the attention computation
        queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Use absolute_position for RoPE offset
        let offset = cache.offset() as i32;
        if offset > 0 {
            queries = self.rope.forward((&queries, offset))?;
            keys = self.rope.forward((&keys, offset))?;
        } else {
            queries = self.rope.forward(&queries)?;
            keys = self.rope.forward(&keys)?;
        }

        // Concatenate with cached keys and values
        let (keys, values) = cache.concatenate(keys, values).map_err(|e| match e {
            crate::models::ModelError::MlxException(ex) => ex,
            _ => mlx_rs::error::Exception::from(e.to_string().as_str()),
        })?;

        let output = scaled_dot_product_attention(queries, keys, values, self.scale, mask)?;
        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        let output = self.o_proj.forward(&output)?;

        Ok(AttentionOutput {
            output,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
    }
}

// ============================================================================
// Mlp (Feed-Forward Network)
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Mlp {
    #[quantizable]
    #[param]
    gate_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    down_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.dim();
        let hidden_dim = args.hidden_dim();
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj: MaybeQuantized::new(gate_proj),
            down_proj: MaybeQuantized::new(down_proj),
            up_proj: MaybeQuantized::new(up_proj),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, x: &'_ Array) -> Result<Self::Output, Self::Error> {
        let w2_input = nn::silu(self.gate_proj.forward(x)?)?.multiply(self.up_proj.forward(x)?)?;
        self.down_proj.forward(&w2_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

// ============================================================================
// ModelLayer (Transformer Block)
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct ModelLayer {
    n_heads: i32,
    dim: i32,

    #[quantizable]
    #[param]
    self_attn: SelfAttn,

    #[quantizable]
    #[param]
    mlp: Mlp,

    #[param]
    input_layernorm: nn::RmsNorm,

    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl ModelLayer {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let n_heads = args.n_heads();
        let dim = args.dim();
        let norm_eps = args.norm_eps();

        let self_attn = SelfAttn::new(args)?;
        let mlp = Mlp::new(args)?;
        let input_layernorm = nn::RmsNormBuilder::new(dim).eps(norm_eps).build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(dim).eps(norm_eps).build()?;
        Ok(Self {
            n_heads,
            dim,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl Module<AttentionInput<'_>> for ModelLayer {
    type Output = AttentionOutput;

    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;
        let norm_x = self.input_layernorm.forward(x)?;
        let attention_input = AttentionInput {
            x: &norm_x,
            mask,
            cache,
        };
        let attention_output = self.self_attn.forward(attention_input)?;

        let r = attention_output.output;

        let h = x.add(r)?;
        let r = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        let output = h.add(r)?;

        Ok(AttentionOutput { output })
    }

    fn training_mode(&mut self, mode: bool) {
        self.self_attn.training_mode(mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

// ============================================================================
// InnerModel (Core Transformer without HuggingFace wrapper)
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct InnerModel {
    vocab_size: i32,
    n_layers: i32,

    #[quantizable]
    #[param]
    embed_tokens: MaybeQuantized<Embedding>,

    #[quantizable]
    #[param]
    layers: Vec<ModelLayer>,

    #[param]
    norm: nn::RmsNorm,
}

impl InnerModel {
    pub fn new(args: &ModelArgs) -> Result<Self, ModelError> {
        let vocab_size = args.vocab_size;
        if vocab_size <= 0 {
            return Err(ModelError::InvalidVocabSize(vocab_size));
        }
        let n_layers = args.n_layers();
        let dim = args.dim();
        let norm_eps = args.norm_eps();

        let embed_tokens = Embedding::new(vocab_size, dim)?;
        let layers = (0..n_layers)
            .map(|_| ModelLayer::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(dim).eps(norm_eps).build()?;

        Ok(Self {
            vocab_size,
            n_layers,
            embed_tokens: MaybeQuantized::new(embed_tokens),
            layers,
            norm,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        self.layers
            .iter_mut()
            .for_each(|layer| layer.training_mode(mode));
        self.norm.training_mode(mode);
    }
}

// ============================================================================
// Model (HuggingFace-compatible wrapper with "model." prefix)
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    #[quantizable]
    #[param]
    model: InnerModel,

    #[quantizable]
    #[param]
    lm_head: MaybeQuantized<nn::Linear>,

    max_position_embeddings: usize,
}

impl Model {
    pub fn new(args: &ModelArgs) -> Result<Self, ModelError> {
        let vocab_size = args.vocab_size;
        if vocab_size <= 0 {
            return Err(ModelError::InvalidVocabSize(vocab_size));
        }
        let dim = args.dim();

        let model = InnerModel::new(args)?;
        let lm_head = nn::LinearBuilder::new(dim, vocab_size)
            .bias(false)
            .build()?;

        Ok(Self {
            model,
            lm_head: MaybeQuantized::new(lm_head),
            max_position_embeddings: args.max_context_size() as usize,
        })
    }
}

// ============================================================================
// Module Implementation for Model
// ============================================================================

impl Module<ModelInput<'_>> for Model {
    type Output = ModelOutput;

    type Error = ModelError;

    fn forward(&mut self, input: ModelInput<'_>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, mut cache } = input;

        let mut h = self.model.embed_tokens.forward(inputs)?;

        let mut mask = None;
        if h.shape()[1] > 1 {
            let mask_ = nn::MultiHeadAttention::create_additive_causal_mask::<f32>(h.shape()[1])?;
            let mask_ = mask_.as_dtype(h.dtype())?;
            mask = Some(mask_);
        }

        // Pass mutable cache references to each layer
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let input = AttentionInput {
                x: &h,
                mask: mask.as_ref().map(Into::into),
                cache: &mut *cache.get_mut(i),
            };
            let output = layer.forward(input)?;
            h = output.output;
        }

        let output = self.lm_head.forward(&self.model.norm.forward(&h)?)?;

        Ok(ModelOutput {
            logits: output,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.model.training_mode(mode);
        self.lm_head.training_mode(mode);
    }
}

// ============================================================================
// LanguageModel Trait Implementation
// ============================================================================

impl LanguageModel for Model {
    fn forward(&mut self, input: ModelInput<'_>) -> Result<ModelOutput, ModelError> {
        <Self as Module<ModelInput<'_>>>::forward(self, input)
    }

    fn max_context_size(&self) -> usize {
        self.max_position_embeddings
    }
}
