"""F04_01: ATR 归一化"""
import numpy as np
from .utils import safe_divide_batch, safe_clip_batch


def calculate(atr_vals, closes):
    """
    ATR 与收盘价比率
    
    Args:
        atr_vals: (batch_size,) ATR值
        closes: (batch_size,) 收盘价数组
    
    Returns:
        (batch_size,) 特征值数组
    """
    atr_norm = safe_divide_batch(atr_vals, closes, 0.0)
    return safe_clip_batch(atr_norm, -1.0, 1.0, 0.0)
