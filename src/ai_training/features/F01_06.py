"""F01_06: MA5趋势斜率"""
import numpy as np
from scipy.stats import linregress


def calculate(idx_array, ma5_prices, atr):
    """
    MA5过去5根K线的趋势斜率（用ATR归一化）
    
    Args:
        idx_array: (batch_size,) 样本索引
        ma5_prices: (total_bars,) MA5价格
        atr: (batch_size,) ATR值
    
    Returns:
        (batch_size,) 斜率特征 [-1, 1]
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
                    
                    slope, _, _, _, _ = linregress(x_valid, y_valid)
                    
                    if np.isfinite(slope):
                        atr_val = atr[i]
                        if atr_val > 1e-8:
                            normalized_slope = np.clip(slope / atr_val, -1.0, 1.0)
                            features[i] = normalized_slope if np.isfinite(normalized_slope) else 0.0
                        else:
                            features[i] = 0.0
                    else:
                        features[i] = 0.0
                else:
                    features[i] = 0.0
            except Exception:
                features[i] = 0.0
        else:
            features[i] = 0.0
    
    return features
