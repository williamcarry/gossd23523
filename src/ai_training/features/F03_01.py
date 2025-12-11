"""F03_01: MA5成交量比率"""
import numpy as np
from .utils import safe_divide_batch


def calculate(ma5_volumes, ma60_volumes):
    """
    MA5 与 MA60 成交量比率
    
    Args:
        ma5_volumes: (batch_size,) MA5成交量
        ma60_volumes: (batch_size,) MA60成交量
    
    Returns:
        (batch_size,) 成交量比率
    """
    vol_ratio = safe_divide_batch(ma5_volumes, ma60_volumes, 1.0)
    vol_ratio = np.where(np.isfinite(vol_ratio), np.minimum(vol_ratio, 3.0), 1.0)
    vol_ratio_norm = (vol_ratio - 1.0) / 2.0
    return np.where(np.isfinite(vol_ratio_norm), vol_ratio_norm, 0.0)
