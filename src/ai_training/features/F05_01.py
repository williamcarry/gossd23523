"""F05_01: 缺口强度"""
import numpy as np


def calculate(idx_array, opens, closes, atr):
    """
    开盘价与前日收盘价的缺口强度
    
    Args:
        idx_array: (batch_size,) 样本索引数组
        opens: (total_bars,) 开盘价
        closes: (total_bars,) 收盘价
        atr: (batch_size,) ATR值
    
    Returns:
        (batch_size,) 缺口强度 [0, 1]
    """
    batch_size = len(idx_array)
    gap_strength = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        if idx > 0:
            prev_close = closes[idx - 1]
            if not np.isfinite(prev_close):
                gap_val = 0.0
            else:
                den = atr[i] if (np.isfinite(atr[i]) and atr[i] > 1e-8) else 1e-8
                raw_gap = np.abs(opens[idx] - prev_close) / den
                gap_val = max(0.0, min(raw_gap / 3.0, 1.0)) if np.isfinite(raw_gap) else 0.0
        else:
            gap_val = 0.0
        gap_strength[i] = gap_val
    
    return gap_strength
