//! Tensor loading utilities

use crate::utils::core_dtype_to_plugin;
use hodu_core::format::hdt;
use hodu_plugin::{PluginDType, TensorData};
use std::path::Path;

/// Maximum file size for tensor loading (100 MB)
const MAX_TENSOR_FILE_SIZE: u64 = 100 * 1024 * 1024;

pub fn load_tensor_file(
    path: &Path,
    expected_shape: &[usize],
    expected_dtype: PluginDType,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    // Check file size before loading
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > MAX_TENSOR_FILE_SIZE {
        return Err(format!(
            "Tensor file too large: {} bytes (max {} MB)",
            metadata.len(),
            MAX_TENSOR_FILE_SIZE / (1024 * 1024)
        )
        .into());
    }

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "hdt" => load_tensor_hdt(path, expected_shape, expected_dtype),
        "json" => load_tensor_json(path, expected_shape, expected_dtype),
        _ => Err(format!("Unsupported tensor format: .{}\nSupported: .hdt, .json", ext).into()),
    }
}

fn load_tensor_hdt(
    path: &Path,
    expected_shape: &[usize],
    expected_dtype: PluginDType,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    let tensor = hdt::load(path).map_err(|e| format!("Failed to load HDT file: {}", e))?;

    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let dtype: PluginDType = core_dtype_to_plugin(tensor.dtype());

    if shape != expected_shape {
        return Err(format!(
            "Shape mismatch: file has {:?}, model expects {:?}",
            shape, expected_shape
        )
        .into());
    }

    if dtype != expected_dtype {
        return Err(format!(
            "DType mismatch: file has {}, model expects {}",
            dtype.name(),
            expected_dtype.name()
        )
        .into());
    }

    let data = tensor
        .to_bytes()
        .map_err(|e| format!("Failed to get tensor bytes: {}", e))?;

    Ok(TensorData::new(data, shape, dtype))
}

fn load_tensor_json(
    path: &Path,
    expected_shape: &[usize],
    expected_dtype: PluginDType,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let obj = json
        .as_object()
        .ok_or("JSON must be an object with shape, dtype, data")?;

    let shape: Vec<usize> = obj
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or("Missing 'shape' field")?
        .iter()
        .map(|v| {
            let n = v.as_u64().ok_or("Invalid shape dimension: expected integer")?;
            usize::try_from(n).map_err(|_| "Shape dimension too large for this platform")
        })
        .collect::<Result<Vec<_>, _>>()?;

    if shape != expected_shape {
        return Err(format!(
            "Shape mismatch: file has {:?}, model expects {:?}",
            shape, expected_shape
        )
        .into());
    }

    let dtype_str = obj
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'dtype' field")?;
    let dtype = str_to_plugin_dtype(dtype_str)?;

    if dtype != expected_dtype {
        return Err(format!(
            "DType mismatch: file has {}, model expects {}",
            dtype.name(),
            expected_dtype.name()
        )
        .into());
    }

    let data_arr = obj
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or("Missing 'data' field")?;

    let data = json_array_to_bytes(data_arr, dtype)?;

    Ok(TensorData::new(data, shape, dtype))
}

pub fn str_to_plugin_dtype(s: &str) -> Result<PluginDType, Box<dyn std::error::Error>> {
    s.parse::<PluginDType>().map_err(|e| e.into())
}

fn json_array_to_bytes(arr: &[serde_json::Value], dtype: PluginDType) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(arr.len() * dtype.size_in_bytes());

    match dtype {
        PluginDType::F32 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")? as f32;
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        },
        PluginDType::F64 => {
            for v in arr {
                let f = v.as_f64().ok_or("Expected number")?;
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        },
        PluginDType::I32 => {
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_i64().ok_or("Expected integer")?;
                let n32 = i32::try_from(n).map_err(|_| format!("Value at index {} ({}) overflows i32", i, n))?;
                bytes.extend_from_slice(&n32.to_le_bytes());
            }
        },
        PluginDType::I64 => {
            for v in arr {
                let n = v.as_i64().ok_or("Expected integer")?;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        PluginDType::U32 => {
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_u64().ok_or("Expected unsigned integer")?;
                let n32 = u32::try_from(n).map_err(|_| format!("Value at index {} ({}) overflows u32", i, n))?;
                bytes.extend_from_slice(&n32.to_le_bytes());
            }
        },
        PluginDType::U64 => {
            for v in arr {
                let n = v.as_u64().ok_or("Expected unsigned integer")?;
                bytes.extend_from_slice(&n.to_le_bytes());
            }
        },
        _ => return Err(format!("Unsupported dtype for JSON input: {}", dtype.name()).into()),
    }

    Ok(bytes)
}
