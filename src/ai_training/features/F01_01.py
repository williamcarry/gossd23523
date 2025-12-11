"""F01_01: MA5价格归一化"""
import numpy as np
from .utils import safe_divide_batch, safe_clip_batch


def calculate(closes, ma5_prices):
    """
    MA5价格归一化
    
    Args:
        closes: (batch_size,) 收盘价数组
        ma5_prices: (batch_size,) MA5价格数组
    
    Returns:
        (batch_size,) 特征值数组
    """
    ma5_norm = safe_divide_batch(ma5_prices, closes, 1.0)
    ma5_norm_scaled = np.where(
        np.isfinite(ma5_norm),
        (ma5_norm - 1.0) * 20,
        0.0
    )
    return safe_clip_batch(ma5_norm_scaled, -1.0, 1.0, 0.0)
