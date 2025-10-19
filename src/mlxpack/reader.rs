/// MLXPack reader - loads model data from single file

use super::{PackMetadata, MAGIC, VERSION, HEADER_SIZE, Result, MLXPackError};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write as IoWrite};
use std::path::Path;
use std::collections::HashMap;
use mlx_rs::Array;

pub struct MLXPackReader {
    file: BufReader<File>,
    metadata: PackMetadata,
    safetensors_data_offset: u64,
}

impl MLXPackReader {
    /// Open an MLXPack file for reading
    pub fn open(pack_path: &Path) -> Result<Self> {
        let file = File::open(pack_path)?;
        let mut reader = BufReader::new(file);

        // Read and validate header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(MLXPackError::InvalidFormat(format!(
                "Invalid magic number: expected {:?}, got {:?}",
                MAGIC, magic
            )));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != VERSION {
            return Err(MLXPackError::UnsupportedVersion(version));
        }

        let mut metadata_size_bytes = [0u8; 8];
        reader.read_exact(&mut metadata_size_bytes)?;
        let metadata_size = u64::from_le_bytes(metadata_size_bytes);

        // Read metadata JSON
        let mut metadata_buffer = vec![0u8; metadata_size as usize];
        reader.read_exact(&mut metadata_buffer)?;
        let metadata: PackMetadata = serde_json::from_slice(&metadata_buffer)?;

        // Calculate where safetensors data starts
        let safetensors_data_offset = HEADER_SIZE as u64 + metadata_size;

        Ok(Self {
            file: reader,
            metadata,
            safetensors_data_offset,
        })
    }

    /// Get the metadata
    pub fn metadata(&self) -> &PackMetadata {
        &self.metadata
    }

    /// Get the model configuration (ModelArgs)
    pub fn config(&self) -> &crate::models::ModelArgs {
        &self.metadata.config
    }

    /// Get the tokenizer.json as raw string
    pub fn tokenizer_json(&self) -> &str {
        &self.metadata.tokenizer_json
    }

    /// Get the tokenizer_config if present
    pub fn tokenizer_config(&self) -> Option<&crate::tokenizer::TokenizerConfig> {
        self.metadata.tokenizer_config.as_ref()
    }

    /// Get the generation_config if present
    pub fn generation_config(&self) -> Option<&crate::tokenizer::GenerationConfig> {
        self.metadata.generation_config.as_ref()
    }

    /// Get the chat_template.jinja content if present
    pub fn chat_template(&self) -> Option<&str> {
        self.metadata.chat_template.as_deref()
    }

    /// Check if the pack has sharded safetensors (multiple files)
    pub fn is_sharded(&self) -> bool {
        self.metadata.safetensors_files.len() > 1
    }

    /// Get list of safetensors filenames
    pub fn safetensors_filenames(&self) -> Vec<&str> {
        self.metadata
            .safetensors_files
            .iter()
            .map(|e| e.name.as_str())
            .collect()
    }

    /// Write a specific safetensors file to disk
    pub fn extract_safetensors_to_file(&mut self, filename: &str, output_path: &Path) -> Result<()> {
        let entry = self
            .metadata
            .safetensors_files
            .iter()
            .find(|e| e.name == filename)
            .ok_or_else(|| MLXPackError::SafetensorsFileNotFound(filename.to_string()))?;

        // Seek to the file's position
        let absolute_offset = self.safetensors_data_offset + entry.offset;
        self.file.seek(SeekFrom::Start(absolute_offset))?;

        // Create output file
        let output_file = File::create(output_path)?;
        let mut writer = BufWriter::new(output_file);

        // Copy data in chunks
        let mut buffer = vec![0u8; 8 * 1024 * 1024]; // 8MB buffer
        let mut remaining = entry.size;

        while remaining > 0 {
            let to_read = std::cmp::min(remaining, buffer.len() as u64) as usize;
            self.file.read_exact(&mut buffer[..to_read])?;
            writer.write_all(&buffer[..to_read])?;
            remaining -= to_read as u64;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load safetensors file directly as MLX Arrays
    /// Returns a HashMap of tensor_name -> Array
    pub fn load_safetensors_arrays(&mut self, filename: &str) -> Result<HashMap<String, Array>> {
        let entry = self
            .metadata
            .safetensors_files
            .iter()
            .find(|e| e.name == filename)
            .ok_or_else(|| MLXPackError::SafetensorsFileNotFound(filename.to_string()))?;

        // Seek to the file's position
        let absolute_offset = self.safetensors_data_offset + entry.offset;
        self.file.seek(SeekFrom::Start(absolute_offset))?;

        // Read the safetensors data into memory
        let mut buffer = vec![0u8; entry.size as usize];
        self.file.read_exact(&mut buffer)?;

        // Create a temporary file to use MLX's load_safetensors
        // (MLX requires a file path, not raw bytes)
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(format!("mlxpack_temp_{}.safetensors", filename.replace('/', "_")));

        std::fs::write(&temp_path, &buffer)?;

        // Load using MLX
        let arrays = Array::load_safetensors(&temp_path)
            .map_err(|e| MLXPackError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("MLX safetensors loading error: {}", e)
            )))?;

        // Clean up temp file
        let _ = std::fs::remove_file(&temp_path);

        Ok(arrays)
    }

    /// Load all safetensors files as MLX Arrays
    /// Returns a HashMap of filename -> (tensor_name -> Array)
    pub fn load_all_safetensors_arrays(&mut self) -> Result<HashMap<String, HashMap<String, Array>>> {
        let filenames: Vec<String> = self.metadata
            .safetensors_files
            .iter()
            .map(|e| e.name.clone())
            .collect();

        let mut all_arrays = HashMap::new();

        for filename in filenames {
            let arrays = self.load_safetensors_arrays(&filename)?;
            all_arrays.insert(filename, arrays);
        }

        Ok(all_arrays)
    }
}
