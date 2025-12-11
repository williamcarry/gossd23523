"""F04_03: 布林带宽度归一化"""
import numpy as np
from .utils import safe_divide_batch, safe_clip_batch


def calculate(upper_bb, lower_bb, atr):
    """
    布林带宽度与ATR的比率
    
    Args:
        upper_bb: (batch_size,) 布林带上轨
        lower_bb: (batch_size,) 布林带下轨
        atr: (batch_size,) ATR值
    
    Returns:
        (batch_size,) 宽度比率 [0, 2]
    """
    band_width = upper_bb - lower_bb
    band_width_norm = safe_divide_batch(band_width, atr, 0.0)
    return safe_clip_batch(band_width_norm, 0.0, 2.0, 0.0)
