use crate::models::{ModelInput, ModelOutput, ModelError, LanguageModel};
use crate::generate::kv_cache::TokensCache;
use mlx_rs::{
    array,
    ops::indexing::{argmax_axis, IndexOp, NewAxis, take_along_axis},
    random::categorical,
    Array,
};
use mlx_rs::ops::{softmax_axis, argsort_axis, cumsum};

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

fn sample(logits: &Array, temp: f32, top_p: f32) -> Result<Array, ModelError> {
    // Greedy decoding when temp is 0
    if temp == 0.0 {
        return Ok(argmax_axis(logits, -1, None)?);
    }

    // Apply temperature scaling
    let logits = logits.multiply(array!(1.0 / temp))?;

    // If top_p is 1.0 or greater, sample from full distribution
    if top_p >= 1.0 {
        return Ok(categorical(&logits, None, None, None)?);
    }

    // Apply nucleus (top-p) sampling
    // 1. Compute softmax probabilities
    let probs = softmax_axis(&logits, -1, None)?;

    // 2. Sort probabilities in DESCENDING order by negating, sorting, then using those indices
    // argsort gives ascending order, so we negate to get descending
    let neg_probs = probs.multiply(array!(-1.0))?;
    let sorted_indices = argsort_axis(&neg_probs, -1)?;

    // 3. Get sorted probabilities (now in descending order)
    let sorted_probs = take_along_axis(&probs, &sorted_indices, -1)?;

    // 4. Compute cumulative sum (from largest to smallest)
    // Use inclusive=false to get cumsum that doesn't include current element yet
    // This way, cumsum[i] = sum of probs[0..i], and we compare to see if adding probs[i] would exceed top_p
    let cumsum_probs = cumsum(&sorted_probs, -1, None, Some(false))?;

    // 5. Create mask: cumsum < top_p means we keep these tokens (nucleus)
    // This keeps tokens while their cumulative sum hasn't exceeded top_p yet
    // At least one token is always kept (even if first prob > top_p, cumsum[0]=0 < top_p)
    let cutoff = array!(top_p);
    let mask = cumsum_probs.lt(&cutoff)?;

    // 6. Create filtered logits array  by setting masked positions to -inf
    let neg_inf = array!(-1e10_f32);

    // Scatter the mask back to original indices using argsort on sorted_indices to get inverse permutation
    let inverse_indices = argsort_axis(&sorted_indices, -1)?;
    let filtered_mask = take_along_axis(&mask, &inverse_indices, -1)?;

    // Apply mask: where mask is true, keep logits; where false, set to -inf
    let logits_dtype = logits.dtype();
    let filtered_logits = filtered_mask
        .as_dtype(logits_dtype)?
        .multiply(&logits)?
        .add(&(!&filtered_mask).as_dtype(logits_dtype)?.multiply(&neg_inf)?)?;

    // 7. Sample from filtered distribution
    Ok(categorical(&filtered_logits, None, None, None)?)
}

pub struct TokensGenerator<'a, M: LanguageModel> {
    model: &'a mut M,
    temp: f32,
    top_p: f32,
    context_size: usize,
    eviction_batch_percent: f32,
    num_sink_tokens: usize,
    state: GenerateTokenState,
}

enum GenerateTokenState {
    Start {
        prompt_token: Array,
    },
    Continue {
        y: Array,
        cache: TokensCache,
    },
}

impl<'a, M: LanguageModel> TokensGenerator<'a, M> {
    pub fn new(
        model: &'a mut M,
        prompt_token: Array,
        temp: f32,
        top_p: f32,
        context_size: usize,
        num_sink_tokens: usize,
        eviction_batch_percent: f32,
    ) -> Self {
        Self {
            model,
            temp,
            top_p,
            context_size,
            num_sink_tokens,
            eviction_batch_percent,
            state: GenerateTokenState::Start { prompt_token },
        }
    }
}

impl<M: LanguageModel> Iterator for TokensGenerator<'_, M> {
    type Item = Result<Array, ModelError>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateTokenState::Start { prompt_token } => {
                let mut cache = TokensCache::new(
                    self.context_size,
                    self.num_sink_tokens,
                    self.eviction_batch_percent
                );

                let input = ModelInput {
                    inputs: prompt_token,
                    cache: cache.clone(),
                };
                let ModelOutput { logits } = tri!(self.model.forward(input));

                // After all layers have processed, truncate all caches synchronously if needed
                tri!(cache.truncate_if_needed());

                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp, self.top_p));

                self.state = GenerateTokenState::Continue {
                    y: y.clone(),
                    cache,
                };

                Some(Ok(y))
            }
            GenerateTokenState::Continue { y, cache } => {
                let next_token = y.index((.., NewAxis));

                let mut cache = cache.clone();
                let input = ModelInput {
                    inputs: &next_token,
                    cache: cache.clone(),
                };
                let ModelOutput { logits } = tri!(self.model.forward(input));

                // After all layers have processed, truncate all caches synchronously if needed
                tri!(cache.truncate_if_needed());

                let logits = tri!(logits.squeeze_axes(&[1]));
                let y = tri!(sample(&logits, self.temp, self.top_p));

                self.state = GenerateTokenState::Continue {
                    y: y.clone(),
                    cache,
                };

                Some(Ok(y))
            }
        }
    }
}