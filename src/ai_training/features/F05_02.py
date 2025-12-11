"""F05_02: 趋势陡峭度"""
import numpy as np


def calculate(idx_array, closes, ma25_prices):
    """
    价格上升与均线的夹角，判断趋势强度
    
    Args:
        idx_array: (batch_size,) 样本索引
        closes: (total_bars,) 收盘价
        ma25_prices: (total_bars,) MA25价格
    
    Returns:
        (batch_size,) 趋势陡峭度 [-1, 1]
    """
    batch_size = len(idx_array)
    trend_angle = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        if idx >= 10:
            if not (np.isfinite(ma25_prices[idx]) and np.isfinite(ma25_prices[idx - 10])):
                trend_val = 0.0
            else:
                price_change = closes[idx] - closes[idx - 10]
                ma25_change = ma25_prices[idx] - ma25_prices[idx - 10]

                if np.abs(ma25_change) < 1e-8:
                    trend_val = 0.0
                else:
                    if np.abs(ma25_change) > 1e-8:
                        angle = price_change / np.abs(ma25_change)
                    else:
                        angle = 0.0
                    trend_val = max(-1.0, min(angle / 2.0, 1.0)) if np.isfinite(angle) else 0.0
        else:
            trend_val = 0.0
        trend_angle[i] = trend_val
    
    return trend_angle
