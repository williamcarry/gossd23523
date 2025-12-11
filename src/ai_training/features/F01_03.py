"""F01_03: MA5>MA25（金叉状态）"""
import numpy as np


def calculate(ma5_prices, ma25_prices):
    """
    MA5 > MA25（金叉状态）
    
    Args:
        ma5_prices: (batch_size,) MA5价格数组
        ma25_prices: (batch_size,) MA25价格数组
    
    Returns:
        (batch_size,) 金叉状态（1.0=是，0.0=否）
    """
    cross_mask = np.isfinite(ma5_prices) & np.isfinite(ma25_prices) & (ma5_prices > ma25_prices)
    return cross_mask.astype(np.float32)
