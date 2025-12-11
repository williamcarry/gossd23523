"""F02_03: MACD柱归一化"""
import numpy as np
from .utils import safe_divide_batch, safe_clip_batch


def calculate(macd_histogram, closes):
    """
    MACD 柱状线归一化
    
    Args:
        macd_histogram: (batch_size,) MACD柱
        closes: (batch_size,) 收盘价
    
    Returns:
        (batch_size,) 特征值
    """
    macd_norm = safe_divide_batch(macd_histogram, closes, 0.0)
    return safe_clip_batch(macd_norm, -0.1, 0.1, 0.0)
