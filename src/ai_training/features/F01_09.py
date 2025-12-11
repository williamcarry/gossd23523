"""F01_09: MA5-MA25粘合度"""
import numpy as np
from .utils import safe_clip_batch


def calculate(ma5_prices, ma25_prices):
    """
    MA5与MA25距离的倒数（粘合度）
    
    Args:
        ma5_prices: (batch_size,) MA5价格
        ma25_prices: (batch_size,) MA25价格
    
    Returns:
        (batch_size,) 粘合度 [0, 1]
    """
    valid_ma_mask = np.isfinite(ma5_prices) & np.isfinite(ma25_prices) & (ma25_prices > 1e-8)
    
    distance = np.full_like(ma5_prices, 1.0, dtype=np.float32)
    distance[valid_ma_mask] = (np.abs(ma5_prices[valid_ma_mask] - ma25_prices[valid_ma_mask]) / 
                               np.maximum(ma25_prices[valid_ma_mask], 1e-8))
    
    distance = safe_clip_batch(distance, 0.0, 1.0, 1.0)
    ma_cohesion = 1.0 - distance
    return np.where(np.isfinite(ma_cohesion), ma_cohesion, 0.0)
