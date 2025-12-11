"""F02_09: MACD柱加速度"""
import numpy as np


def calculate(idx_array, macd_histogram, closes):
    """
    MACD柱线的加速度（二阶导数）
    
    Args:
        idx_array: (batch_size,) 样本索引
        macd_histogram: (total_bars,) MACD柱值
        closes: (total_bars,) 收盘价
    
    Returns:
        (batch_size,) 加速度 [-0.1, 0.1]
    """
    batch_size = len(idx_array)
    features = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        if idx >= 2:
            macd_curr = macd_histogram[idx]
            macd_prev = macd_histogram[idx-1]
            macd_prev_prev = macd_histogram[idx-2]
            close_curr = closes[idx]
            
            if (np.isfinite(macd_curr) and np.isfinite(macd_prev) and 
                np.isfinite(macd_prev_prev) and np.isfinite(close_curr)):
                accel = (macd_curr - macd_prev) - (macd_prev - macd_prev_prev)
                accel_norm = accel / (close_curr + 1e-8) if np.isfinite(accel) and np.isfinite(close_curr) else 0.0
                accel = np.clip(accel_norm, -0.1, 0.1) if np.isfinite(accel_norm) else 0.0
                features[i] = accel
            else:
                features[i] = 0.0
        else:
            features[i] = 0.0
    
    return features
