"""F02_02: MACD DEA归一化"""
import numpy as np
from .utils import safe_divide_batch, safe_clip_batch


def calculate(dea_vals, closes):
    """
    MACD DEA 归一化
    
    Args:
        dea_vals: (batch_size,) DEA数组
        closes: (batch_size,) 收盘价数组
    
    Returns:
        (batch_size,) 特征值数组
    """
    dea_norm = safe_divide_batch(dea_vals, closes, 0.0)
    return safe_clip_batch(dea_norm, -0.1, 0.1, 0.0)
