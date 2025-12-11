"""F02_01: MACD DIF归一化"""
import numpy as np
from .utils import safe_divide_batch, safe_clip_batch


def calculate(dif_vals, closes):
    """
    MACD DIF 归一化
    
    Args:
        dif_vals: (batch_size,) DIF数组
        closes: (batch_size,) 收盘价数组
    
    Returns:
        (batch_size,) 特征值数组
    """
    dif_norm = safe_divide_batch(dif_vals, closes, 0.0)
    return safe_clip_batch(dif_norm, -0.1, 0.1, 0.0)
