use crate::models::ModelError;
use mlx_rs::Array;
use mlx_rs::ops::{concatenate_axis, indexing::IndexOp};
use std::cell::{RefCell, RefMut};
use std::ops::DerefMut;
use std::rc::Rc;

/// Manages a collection of KeyValueCache instances, one per model layer.
/// Lazily initializes caches as layers request them.
/// Uses Rc<RefCell<>> internally so cloning is cheap and shares the same underlying data.
#[derive(Clone)]
pub struct TokensCache {
    caches: Rc<RefCell<Vec<KeyValueCache>>>,
    max_context_size: usize,
    num_sink_tokens: usize,
    eviction_batch_percent: f32,
}

impl TokensCache {
    pub fn new(
        max_context_size: usize,
        num_sink_tokens: usize,
        eviction_batch_percent: f32,
    ) -> Self {
        Self {
            caches: Rc::new(RefCell::new(Vec::new())),
            max_context_size,
            num_sink_tokens,
            eviction_batch_percent: eviction_batch_percent.clamp(0.01, 0.5),
        }
    }

    /// Get mutable reference to cache for a specific layer, creating it if needed
    pub fn get_mut(&mut self, layer_idx: usize) -> impl DerefMut<Target = KeyValueCache> + '_ {
        // First ensure we have enough caches
        {
            let mut caches = self.caches.borrow_mut();
            while caches.len() <= layer_idx {
                caches.push(KeyValueCache::new(self.num_sink_tokens));
            }
        }
        // Then return the mutable reference
        RefMut::map(self.caches.borrow_mut(), |caches| &mut caches[layer_idx])
    }

    /// Get the number of initialized caches
    pub fn len(&self) -> usize {
        self.caches.borrow().len()
    }

    /// Truncate all layer caches using batch eviction strategy.
    /// This should be called after all layers have processed a token to ensure
    /// all layers maintain the same sequence length.
    ///
    /// Uses configurable batch eviction: when cache exceeds max_context_size,
    /// evicts a percentage of the cache at once (default 10%) to create room
    /// for future tokens, reducing the frequency of eviction operations.
    pub fn truncate_if_needed(&mut self) -> Result<(), ModelError> {
        let mut caches = self.caches.borrow_mut();

        if caches.is_empty() {
            return Ok(());
        }

        // Check if any cache exceeds the limit
        let mut needs_truncation = false;
        let mut max_seq_len = 0;
        for cache in caches.iter() {
            let seq_len = cache.sequence_length();
            max_seq_len = max_seq_len.max(seq_len);
            if seq_len > self.max_context_size {
                needs_truncation = true;
            }
        }

        if needs_truncation {
            // Calculate batch eviction size: percentage of max_context_size
            // This creates "room" so we don't evict on every single token
            let batch_size =
                (self.max_context_size as f32 * self.eviction_batch_percent).ceil() as usize;
            let batch_size = batch_size.max(1); // Ensure at least 1 token is evicted

            // Evict enough tokens to: (1) get back to max_size, (2) create room for next batch_size tokens
            let tokens_to_remove = (max_seq_len - self.max_context_size) + batch_size;
            let target_size = max_seq_len - tokens_to_remove;

            eprintln!(
                "[INFO] StreamingLLM: Context limit ({}) exceeded. Batch evicting {} middle tokens ({}% of cache) from ALL layers (seq_len: {} -> {}, creating room for {} more tokens)",
                self.max_context_size,
                tokens_to_remove,
                (self.eviction_batch_percent * 100.0) as usize,
                max_seq_len,
                target_size,
                batch_size
            );

            // Evict middle tokens from all caches synchronously using StreamingLLM pattern
            // This preserves attention sinks at the start and recent tokens at the end
            for cache in caches.iter_mut() {
                cache.evict_middle_tokens(tokens_to_remove)?;
            }
        }

        Ok(())
    }
}

/// Manages Key-Value cache for a single transformer layer using StreamingLLM pattern.
///
/// StreamingLLM Pattern:
/// - Preserves first `num_sink_tokens` as "attention sinks" (critical initial tokens)
/// - Maintains a sliding window of recent tokens
/// - Evicts middle tokens in batches when cache reaches capacity
///
/// This prevents model degeneration by keeping the attention sink tokens that models
/// rely on for stable generation, even when they're not semantically important.
pub struct KeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    /// Number of initial tokens to preserve as attention sinks (typically 4)
    num_sink_tokens: usize,
    /// Tracks the absolute position in the sequence (for RoPE offset calculation).
    /// This is different from sequence_length() after truncation occurs.
    offset: usize,
}

impl KeyValueCache {
    /// Create a new cache with full configuration
    ///
    /// # Arguments
    /// * `max_context_size` - Maximum number of tokens to cache
    /// * `num_sink_tokens` - Number of initial tokens to preserve as attention sinks (typically 4)
    /// * `eviction_batch_percent` - Percentage of cache to evict at once (0.0-1.0), default 0.1 (10%)
    pub fn new(num_sink_tokens: usize) -> Self {
        Self {
            keys: None,
            values: None,
            num_sink_tokens,
            offset: 0,
        }
    }

    /// Get the current cache for reading
    pub fn get(&self) -> Option<(&Array, &Array)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Get the current sequence length (number of tokens in cache)
    pub fn sequence_length(&self) -> usize {
        if let Some(keys) = &self.keys {
            keys.shape()[2] as usize // keys shape is [batch, n_heads, seq_len, head_dim]
        } else {
            0
        }
    }

    /// Get the absolute position in the sequence for RoPE offset calculation.
    /// This tracks the total number of tokens seen, even after truncation.
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn concatenate(
        &mut self,
        new_keys: Array,
        new_values: Array,
    ) -> Result<(&Array, &Array), ModelError> {
        // Track how many new tokens we're adding
        let new_token_count = new_keys.shape()[2] as usize;

        match (&self.keys, &self.values) {
            (Some(old_keys), Some(old_values)) => {
                // Concatenate old with new (no truncation here)
                self.keys = Some(concatenate_axis(&[old_keys, &new_keys], 2)?);
                self.values = Some(concatenate_axis(&[old_values, &new_values], 2)?);
            }
            _ => {
                // First time, just store (no concatenation needed)
                self.keys = Some(new_keys);
                self.values = Some(new_values);
            }
        }

        // Update absolute position
        self.offset += new_token_count;

        Ok((self.keys.as_ref().unwrap(), self.values.as_ref().unwrap()))
    }

    /// Evict tokens from the middle using StreamingLLM pattern.
    /// Preserves attention sink tokens at the start and recent tokens at the end.
    ///
    /// Layout after eviction:
    /// [sink_tokens][...evicted...][recent_tokens]
    ///  preserved
    pub fn evict_middle_tokens(&mut self, tokens_to_remove: usize) -> Result<(), ModelError> {
        if tokens_to_remove == 0 {
            return Ok(());
        }

        match (&self.keys, &self.values) {
            (Some(keys), Some(values)) => {
                let seq_len = keys.shape()[2] as usize;

                // Ensure we have enough tokens to evict from the middle
                if seq_len <= self.num_sink_tokens {
                    // Not enough tokens yet to have sinks + middle, don't evict
                    return Ok(());
                }

                if tokens_to_remove >= seq_len - self.num_sink_tokens {
                    // Would remove too many tokens, keep only sinks
                    let sink_end = self.num_sink_tokens as i32;
                    let new_keys = keys.index((.., .., ..sink_end, ..));
                    let new_values = values.index((.., .., ..sink_end, ..));
                    self.keys = Some(new_keys);
                    self.values = Some(new_values);
                } else {
                    // StreamingLLM eviction: keep [sinks][...evict...][recent]
                    // 1. Extract sink tokens [0..num_sink_tokens]
                    let sink_end = self.num_sink_tokens as i32;
                    let sink_keys = keys.index((.., .., ..sink_end, ..));
                    let sink_values = values.index((.., .., ..sink_end, ..));

                    // 2. Calculate where to keep recent tokens from
                    // We want to remove tokens_to_remove from the middle
                    // So we keep: sinks + (seq_len - num_sinks - tokens_to_remove) recent
                    let recent_start = (self.num_sink_tokens + tokens_to_remove) as i32;
                    let recent_keys = keys.index((.., .., recent_start.., ..));
                    let recent_values = values.index((.., .., recent_start.., ..));

                    // 3. Concatenate sinks + recent
                    let new_keys = concatenate_axis(&[&sink_keys, &recent_keys], 2)?;
                    let new_values = concatenate_axis(&[&sink_values, &recent_values], 2)?;

                    self.keys = Some(new_keys);
                    self.values = Some(new_values);
                }
                Ok(())
            }
            _ => Ok(()), // Nothing to evict
        }
    }
}
