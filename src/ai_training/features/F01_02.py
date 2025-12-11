"""F01_02: MA25价格归一化"""
import numpy as np
from .utils import safe_divide_batch, safe_clip_batch


def calculate(closes, ma25_prices):
    """
    MA25价格归一化
    
    Args:
        closes: (batch_size,) 收盘价数组
        ma25_prices: (batch_size,) MA25价格数组
    
    Returns:
        (batch_size,) 特征值数组
    """
    ma25_norm = safe_divide_batch(ma25_prices, closes, 1.0)
    ma25_norm_scaled = np.where(
        np.isfinite(ma25_norm),
        (ma25_norm - 1.0) * 10,
        0.0
    )
    return safe_clip_batch(ma25_norm_scaled, -1.0, 1.0, 0.0)
