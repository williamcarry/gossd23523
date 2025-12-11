"""
共享工具函数 - 所有特征码使用
"""

import numpy as np
from scipy.stats import linregress


def safe_divide_batch(numerator, denominator, default=1.0, eps=1e-8):
    """
    ✅ 批量安全除法 - 完整的 NaN/Inf 防护
    
    处理以下边界情况：
    1. NaN / 任何数 = default
    2. 任何数 / NaN = default
    3. 任何数 / 0 = default
    4. 任何数 / 极小值 = default
    5. Inf / 任何数 = default
    """
    if np.isscalar(numerator):
        numerator = np.full_like(denominator, numerator, dtype=np.float32)
    if np.isscalar(denominator):
        denominator = np.full_like(numerator, denominator, dtype=np.float32)

    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)

    valid_mask = (
        np.isfinite(numerator) &
        np.isfinite(denominator) &
        (np.abs(denominator) >= eps)
    )

    result = np.full_like(numerator, default, dtype=np.float32)
    result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    inf_mask = ~np.isfinite(result)
    if np.any(inf_mask):
        result[inf_mask] = default

    return result


def safe_clip_batch(arr, min_val, max_val, default=0.0):
    """
    ✅ 批量安全 Clip 操作
    """
    arr = np.asarray(arr, dtype=np.float32)
    result = np.full_like(arr, default, dtype=np.float32)
    
    valid_mask = np.isfinite(arr)
    result[valid_mask] = np.clip(arr[valid_mask], min_val, max_val)
    
    return result
