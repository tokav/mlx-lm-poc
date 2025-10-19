use mlx_macros::ModuleParameters;
use mlx_rs::{
    Array,
    error::Exception,
    module::{Module, Param},
    ops::{dequantize, quantized_matmul},
    quantization::Quantizable,
};
use mlx_rs::ops::indexing::IndexOp;

// ============================================================================
// Embedding wrapper - necessary to implement Quantizable trait
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct Embedding {
    /// The weight of the
    #[param]
    pub weight: Param<Array>,
}

impl Embedding {
    pub fn new(embedding_count: i32, dimensions: i32) -> Result<Self, Exception> {
        let scale = f32::sqrt(1.0 / (dimensions as f32));
        let weight =
            mlx_rs::random::normal::<f32>(&[embedding_count, dimensions], None, None, None)?
                * scale;

        Ok(Self {
            weight: Param::new(weight),
        })
    }
    pub fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        mlx_rs::ops::matmul(x, self.weight.value.t())
    }
}

impl Quantizable for Embedding {
    type Quantized = QuantizedEmbedding;
    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        QuantizedEmbedding::from_embedding(self, group_size, bits)
    }
}

impl Module<&Array> for Embedding {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        Ok(self.weight.index(x))
    }

    fn training_mode(&mut self, _mode: bool) {}
}

// ============================================================================
// QuantizedEmbedding - Fixed implementation with proper #[param] attributes
// ============================================================================

/// Custom QuantizedEmbedding with proper parameter exposure
///
/// This fixes the bug in MLX-RS 0.25.2 where QuantizedEmbedding
/// doesn't expose its parameters because the #[param] attributes are missing.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedEmbedding {
    /// Quantization group size
    pub group_size: i32,

    /// Bits per parameter
    pub bits: i32,

    /// Scales
    #[param]
    pub scales: Param<Array>,

    /// Biases
    #[param]
    pub biases: Param<Array>,

    /// Inner embedding
    #[param]
    pub inner: mlx_rs::nn::Embedding,
}

impl QuantizedEmbedding {
    /// Default group size
    pub const DEFAULT_GROUP_SIZE: i32 = 64;

    /// Default bits
    pub const DEFAULT_BITS: i32 = 4;

    /// Create a new QuantizedEmbedding from our custom Embedding wrapper
    pub fn from_embedding(
        embedding: Embedding,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        let weight = embedding.weight.value;
        Self::from_weight(weight, group_size, bits)
    }

    /// Create a new QuantizedEmbedding from a weight matrix
    pub fn from_weight(weight: Array, group_size: i32, bits: i32) -> Result<Self, Exception> {
        let (quantized_weight, scales, biases) = mlx_rs::ops::quantize(&weight, group_size, bits)?;

        let inner = mlx_rs::nn::Embedding {
            weight: Param::new(quantized_weight),
        };

        Ok(Self {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner,
        })
    }

    pub fn as_linear(&self, x: impl AsRef<Array>) -> Result<Array, Exception> {
        quantized_matmul(
            x.as_ref(),
            &self.inner.weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )
    }
}

// ModuleParameters is derived automatically

impl Module<&Array> for QuantizedEmbedding {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        use mlx_rs::ops::indexing::IndexOp;
        use std::iter::once;

        let s = x.shape();
        let x = x.flatten(None, None)?;
        let w = self.inner.weight.index(&x);
        let scales = self.scales.index(&x);
        let biases = self.biases.index(&x);

        let out = dequantize(&w, &scales, &biases, self.group_size, self.bits)?;

        let ret_shape = s.iter().copied().chain(once(-1)).collect::<Vec<_>>();
        out.reshape(&ret_shape)
    }

    fn training_mode(&mut self, mode: bool) {
        self.inner.training_mode(mode);
    }
}
