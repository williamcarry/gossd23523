"""F01_10: MA5-MA25发散速度"""
import numpy as np


def calculate(idx_array, ma5_prices, ma25_prices):
    """
    MA5与MA25粘合度的变化速度（指数加权）
    
    Args:
        idx_array: (batch_size,) 样本索引
        ma5_prices: (total_bars,) MA5价格
        ma25_prices: (total_bars,) MA25价格
    
    Returns:
        (batch_size,) 发散速度
    """
    batch_size = len(idx_array)
    features = np.zeros(batch_size, dtype=np.float32)
    
    # 先计算当前粘合度
    valid_ma_mask = np.isfinite(ma5_prices) & np.isfinite(ma25_prices) & (ma25_prices > 1e-8)
    ma_cohesion = np.ones(len(ma5_prices), dtype=np.float32)
    
    distance = np.ones(len(ma5_prices), dtype=np.float32)
    distance[valid_ma_mask] = (np.abs(ma5_prices[valid_ma_mask] - ma25_prices[valid_ma_mask]) / 
                               np.maximum(ma25_prices[valid_ma_mask], 1e-8))
    distance = np.clip(distance, 0.0, 1.0)
    ma_cohesion = 1.0 - distance
    
    for i, idx in enumerate(idx_array):
        if idx >= 10:
            try:
                hist_indices = np.arange(idx-10, idx)
                hist_ma5 = ma5_prices[hist_indices]
                hist_ma25 = ma25_prices[hist_indices]
                
                valid_mask = np.isfinite(hist_ma5) & np.isfinite(hist_ma25) & (hist_ma25 > 1e-8)
                if np.sum(valid_mask) >= 2:
                    valid_ma5 = hist_ma5[valid_mask]
                    valid_ma25 = hist_ma25[valid_mask]
                    
                    hist_distance = np.abs(valid_ma5 - valid_ma25) / np.maximum(valid_ma25, 1e-8)
                    hist_cohesion = 1.0 - np.minimum(hist_distance, 1.0)
                    
                    weights = np.exp(-0.2 * np.arange(len(hist_cohesion))[::-1])
                    weights /= weights.sum()
                    divergence_speed = np.sum(weights * hist_cohesion)
                    
                    current_cohesion = ma_cohesion[idx]
                    if np.isfinite(current_cohesion) and np.isfinite(divergence_speed):
                        features[i] = current_cohesion - divergence_speed
                    else:
                        features[i] = 0.0
                else:
                    features[i] = 0.0
            except Exception:
                features[i] = 0.0
        else:
            features[i] = 0.0
    
    return features
