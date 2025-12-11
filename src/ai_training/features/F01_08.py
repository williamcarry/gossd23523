"""F01_08: K线方向强度"""
import numpy as np
from .utils import safe_divide_batch


def calculate(closes, opens, highs, lows):
    """
    K线方向强度 = 实体大小 * 方向
    
    Args:
        closes: (batch_size,) 收盘价
        opens: (batch_size,) 开盘价
        highs: (batch_size,) 最高价
        lows: (batch_size,) 最低价
    
    Returns:
        (batch_size,) K线强度 [-1, 1]
    """
    body_size = np.abs(closes - opens)
    full_range = highs - lows
    k_line_strength = safe_divide_batch(body_size, full_range, 0.0)
    
    direction = np.where(
        np.isfinite(closes) & np.isfinite(opens),
        np.where(closes < opens, -1.0, 1.0),
        0.0
    )
    k_line_strength = k_line_strength * direction
    return np.where(np.isfinite(k_line_strength), k_line_strength, 0.0)
