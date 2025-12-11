"""F02_06: DIF变化率"""
import numpy as np
from .utils import safe_divide_batch


def calculate(idx_array, dif, closes):
    """
    DIF相对于收盘价的变化率
    
    Args:
        idx_array: (batch_size,) 样本索引
        dif: (total_bars,) DIF值
        closes: (total_bars,) 收盘价
    
    Returns:
        (batch_size,) 变化率 [-0.05, 0.05]
    """
    batch_size = len(idx_array)
    features = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        if idx > 0:
            dif_curr = dif[idx]
            dif_prev = dif[idx-1]
            close_curr = closes[idx]
            
            if np.isfinite(dif_curr) and np.isfinite(dif_prev) and np.isfinite(close_curr):
                dif_change = safe_divide_batch(dif_curr - dif_prev, close_curr, 0.0)
                dif_change = np.clip(dif_change, -0.05, 0.05) if np.isfinite(dif_change) else 0.0
                features[i] = dif_change
            else:
                features[i] = 0.0
        else:
            features[i] = 0.0
    
    return features
