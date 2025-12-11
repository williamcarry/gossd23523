"""F04_02: 布林带位置"""
import numpy as np
from .utils import safe_divide_batch, safe_clip_batch


def calculate(closes, upper_bb, lower_bb):
    """
    价格在布林带中的相对位置
    
    Args:
        closes: (batch_size,) 收盘价
        upper_bb: (batch_size,) 布林带上轨
        lower_bb: (batch_size,) 布林带下轨
    
    Returns:
        (batch_size,) 布林带位置 [0, 1]
    """
    band_width = upper_bb - lower_bb
    valid_mask = (np.isfinite(upper_bb) & np.isfinite(lower_bb) & 
                  np.isfinite(band_width) & (band_width > 1e-8))

    bollinger_position = np.full(len(closes), 0.5, dtype=np.float32)
    
    if np.sum(valid_mask) > 0:
        bollinger_position[valid_mask] = safe_divide_batch(
            closes[valid_mask] - lower_bb[valid_mask],
            band_width[valid_mask],
            0.5
        )
    
    return safe_clip_batch(bollinger_position, 0.0, 1.0, 0.5)
