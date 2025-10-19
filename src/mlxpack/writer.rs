/// MLXPack writer - packs HuggingFace model directory into single file

use super::{PackMetadata, MAGIC, VERSION, Result, MLXPackError};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write, Read};
use std::path::Path;

pub struct MLXPackWriter {
    metadata: PackMetadata,
    model_path: std::path::PathBuf,
}

impl MLXPackWriter {
    /// Create a new writer for packing a model directory
    pub fn new(model_path: &Path) -> Result<Self> {
        let metadata = PackMetadata::from_model_dir(model_path)?;
        Ok(Self {
            metadata,
            model_path: model_path.to_path_buf(),
        })
    }

    /// Pack the model into a single MLXPack file
    pub fn pack(&mut self, output_path: &Path) -> Result<()> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // Step 1: Calculate offsets for safetensors files
        self.calculate_offsets()?;

        // Step 2: Serialize metadata to get its size
        let metadata_json = serde_json::to_vec(&self.metadata)?;
        let metadata_size = metadata_json.len() as u64;

        // Step 3: Write header
        writer.write_all(&MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&metadata_size.to_le_bytes())?;

        // Step 4: Write metadata JSON
        writer.write_all(&metadata_json)?;

        // Step 5: Write safetensors files
        self.write_safetensors_files(&mut writer)?;

        writer.flush()?;

        Ok(())
    }

    /// Calculate byte offsets for each safetensors file
    fn calculate_offsets(&mut self) -> Result<()> {
        let mut current_offset = 0u64;

        for entry in &mut self.metadata.safetensors_files {
            entry.offset = current_offset;
            current_offset += entry.size;
        }

        Ok(())
    }

    /// Write all safetensors files to the pack
    fn write_safetensors_files<W: Write>(&self, writer: &mut W) -> Result<()> {
        for entry in &self.metadata.safetensors_files {
            let file_path = self.model_path.join(&entry.name);
            let mut file = BufReader::new(File::open(&file_path)?);

            // Copy file contents with buffering
            let mut buffer = vec![0u8; 8 * 1024 * 1024]; // 8MB buffer
            let mut total_written = 0u64;

            loop {
                let bytes_read = file.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }

                writer.write_all(&buffer[..bytes_read])?;
                total_written += bytes_read as u64;
            }

            // Verify we wrote the expected amount
            if total_written != entry.size {
                return Err(MLXPackError::InvalidFormat(format!(
                    "File '{}': expected {} bytes, wrote {} bytes",
                    entry.name, entry.size, total_written
                )));
            }

            eprintln!("[INFO] Packed: {} ({} bytes)", entry.name, entry.size);
        }

        Ok(())
    }

    /// Get the metadata (useful for inspection before packing)
    pub fn metadata(&self) -> &PackMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use crate::mlxpack::SafetensorsFileEntry;

    #[test]
    fn test_offset_calculation() {
        let mut safetensors_files = vec![
            SafetensorsFileEntry {
                name: "file1.safetensors".to_string(),
                offset: 0,
                size: 1000,
            },
            SafetensorsFileEntry {
                name: "file2.safetensors".to_string(),
                offset: 0,
                size: 2000,
            },
        ];

        let mut current_offset = 0u64;
        for entry in &mut safetensors_files {
            entry.offset = current_offset;
            current_offset += entry.size;
        }

        assert_eq!(safetensors_files[0].offset, 0);
        assert_eq!(safetensors_files[1].offset, 1000);
    }
}
