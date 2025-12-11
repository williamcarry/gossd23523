"""F01_07: MA5稳定性（R²值）"""
import numpy as np
from scipy.stats import linregress


def calculate(idx_array, ma5_prices):
    """
    MA5过去5根K线回归的R²值（稳定性）
    
    Args:
        idx_array: (batch_size,) 样本索引
        ma5_prices: (total_bars,) MA5价格
    
    Returns:
        (batch_size,) R²值 [0, 1]
    """
    batch_size = len(idx_array)
    features = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        if idx >= 5:
            try:
                ma5_window = ma5_prices[idx-4:idx+1]
                x = np.arange(5, dtype=np.float32)
                
                valid_mask = np.isfinite(ma5_window)
                if np.sum(valid_mask) >= 3:
                    x_valid = x[valid_mask]
                    y_valid = ma5_window[valid_mask]
                    
                    _, _, r_value, _, _ = linregress(x_valid, y_valid)
                    
                    if np.isfinite(r_value):
                        features[i] = np.clip(r_value ** 2, 0.0, 1.0)
                    else:
                        features[i] = 0.0
                else:
                    features[i] = 0.0
            except Exception:
                features[i] = 0.0
        else:
            features[i] = 0.0
    
    return features
