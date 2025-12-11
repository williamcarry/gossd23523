"""
波动率特征组（F04_01~F04_03）

本模块包含所有与波动率相关的3个特征计算
从 feature_extractor.py 提取，保持100%一致
"""
import numpy as np
from .config import EPS
from .utils import safe_divide, compute_atr_with_momentum


def extract_f04_features(
    idx, close,
    atr, upper_bb, middle_bb, lower_bb
):
    """
    提取波动率特征组（F04_01~F04_03）
    
    参数:
        idx: 当前K线索引
        close: 当前收盘价
        atr: ATR数组
        upper_bb: 布林带上轨数组
        middle_bb: 布林带中轨数组
        lower_bb: 布林带下轨数组
    
    返回:
        list: 包含3个特征值的列表 [F04_01, F04_02, F04_03]
    """
    features = []
    
    atr_val = atr[idx]

    # === F04_01: ATR融合(动量+水平) ===
    atr_combined = compute_atr_with_momentum(atr, idx, short_period=5)
    # ✅ NaN防护：检查返回值有效性
    if not np.isfinite(atr_combined):
        atr_combined = 0.0
    features.append(atr_combined)  # F04_01
    
    # === F04_02: 布林带位置 ===
    # ✅ NaN防护：先检查上下轨的有效性
    upper_bb_val = upper_bb[idx]
    lower_bb_val = lower_bb[idx]
    if not (np.isfinite(upper_bb_val) and np.isfinite(lower_bb_val)):
        bollinger_position = 0.5
    else:
        band_width = upper_bb_val - lower_bb_val
        if band_width > 0:
            bollinger_position = safe_divide(close - lower_bb_val, band_width, 0.5)
            # ✅ 修复NaN传播：np.clip会传播NaN，改用max/min
            bollinger_position = max(0.0, min(bollinger_position, 1.0)) if np.isfinite(bollinger_position) else 0.5
        else:
            bollinger_position = 0.5
    features.append(bollinger_position)  # F04_02

    # === F04_03: 布林带宽度归一化 ===
    # 修复：对极低价股增加保护，middle_bb过小时使用ATR作为分母
    # ✅ NaN防护：先检查 band_width 和 middle_bb 的有效性
    middle_bb_val = middle_bb[idx]
    if not np.isfinite(middle_bb_val):
        # middle_bb 无效时，使用ATR作为分母
        if not np.isfinite(upper_bb_val) or not np.isfinite(lower_bb_val):
            band_width_norm = 0.0
        else:
            band_width = upper_bb_val - lower_bb_val
            if not np.isfinite(band_width) or band_width < 0:
                band_width_norm = 0.0
            else:
                den = atr_val if (np.isfinite(atr_val) and atr_val > EPS) else EPS
                band_width_norm = safe_divide(band_width, den, 0)
    elif middle_bb_val > EPS:
        # middle_bb 有效且 > EPS
        if not np.isfinite(upper_bb_val) or not np.isfinite(lower_bb_val):
            band_width_norm = 0.0
        else:
            band_width = upper_bb_val - lower_bb_val
            if not np.isfinite(band_width) or band_width < 0:
                band_width_norm = 0.0
            else:
                band_width_norm = safe_divide(band_width, middle_bb_val, 0)
    else:
        # middle_bb 有效但 <= EPS，使用ATR作为分母
        if not np.isfinite(upper_bb_val) or not np.isfinite(lower_bb_val):
            band_width_norm = 0.0
        else:
            band_width = upper_bb_val - lower_bb_val
            if not np.isfinite(band_width) or band_width < 0:
                band_width_norm = 0.0
            else:
                den = atr_val if (np.isfinite(atr_val) and atr_val > EPS) else EPS
                band_width_norm = safe_divide(band_width, den, 0)
    # ✅ 修复NaN传播：最终检查 band_width_norm 有效性
    band_width_norm = max(-1.0, min(band_width_norm, 1.0)) if np.isfinite(band_width_norm) else 0.0
    features.append(band_width_norm)  # F04_03
    
    return features


# 导出
__all__ = ['extract_f04_features']
