"""F02_07: DEA变化率"""
import numpy as np
from .utils import safe_divide_batch


def calculate(idx_array, dea, closes):
    """
    DEA相对于收盘价的变化率
    
    Args:
        idx_array: (batch_size,) 样本索引
        dea: (total_bars,) DEA值
        closes: (total_bars,) 收盘价
    
    Returns:
        (batch_size,) 变化率 [-0.05, 0.05]
    """
    batch_size = len(idx_array)
    features = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        if idx > 0:
            dea_curr = dea[idx]
            dea_prev = dea[idx-1]
            close_curr = closes[idx]
            
            if np.isfinite(dea_curr) and np.isfinite(dea_prev) and np.isfinite(close_curr):
                dea_change = safe_divide_batch(dea_curr - dea_prev, close_curr, 0.0)
                dea_change = np.clip(dea_change, -0.05, 0.05) if np.isfinite(dea_change) else 0.0
                features[i] = dea_change
            else:
                features[i] = 0.0
        else:
            features[i] = 0.0
    
    return features
