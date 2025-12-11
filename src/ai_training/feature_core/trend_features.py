"""
趋势特征组（F05_01~F05_02）

本模块包含所有与趋势相关的2个特征计算
从 feature_extractor.py 提取，保持100%一致
"""
import numpy as np
from .config import EPS
from .utils import safe_divide


def extract_f05_features(
    idx, open_price, closes,
    atr, ma25_prices
):
    """
    提取趋势特征组（F05_01~F05_02）
    
    参数:
        idx: 当前K线索引
        open_price: 当前开盘价
        closes: 收盘价数组
        atr: ATR数组
        ma25_prices: MA25数组
    
    返回:
        list: 包含2个特征值的列表 [F05_01, F05_02]
    """
    features = []
    
    atr_val = atr[idx]
    ma25 = ma25_prices[idx]
    prev_close = closes[idx-1] if idx > 0 else closes[idx]
    
    # === F05_01: 缺口强度 ===
    # 定义：开盘价与前日收盘价的缺口，判断跳空强度
    # ✅ P0修复：使用ATR归一化，避免极端值
    #    - 原实现：gap/close 可达0.2（涨停）→ 远超[-1,1]范围
    #    - 修复后：gap/ATR 映射到[-1,1]，3倍ATR为满值
    # ✅ P1修复：max(EPS, atr_val)在atr_val=NaN时返回NaN，改用显式检查
    if idx > 0:
        # ✅ NaN防护：先检查 prev_close 是否有效
        if not np.isfinite(prev_close):
            gap_strength = 0.0
        else:
            # 先检查ATR有限性，避免NaN传播
            den = atr_val if (np.isfinite(atr_val) and atr_val > EPS) else EPS
            raw_gap = safe_divide(abs(open_price - prev_close), den, 0)
            # 3倍ATR映射为1.0（经验阈值：缺口>3倍ATR视为极强跳空）
            # ✅ 修复NaN传播：np.clip会传播NaN，改用max/min
            gap_strength = max(0.0, min(raw_gap / 3.0, 1.0)) if np.isfinite(raw_gap) else 0.0
    else:
        gap_strength = 0.0
    features.append(gap_strength)  # F05_01
    
    # === F05_02: 趋势陡峭度 ===
    # 定义：价格上升与均线的夹角，判断趋势强度和可持续性
    # ✅ Bug#8修复：MA25横盘时返回0，避免分母极小值
    if idx >= 10:
        # ✅ NaN防护：检查 ma25 和 ma25_prices[idx-10] 的有效性
        if not (np.isfinite(ma25) and np.isfinite(ma25_prices[idx - 10])):
            trend_angle_normalized = 0.0
        else:
            # 向量化计算：过去10根K线的价格和均线变化
            price_change = closes[idx] - closes[idx - 10]
            ma25_change = ma25 - ma25_prices[idx - 10]

            # ✅ Bug#8修复：MA25横盘时直接返回0
            if abs(ma25_change) < EPS:
                trend_angle_normalized = 0.0
            else:
                # 夹角：价格变化速度 / 均线变化速度
                trend_angle = safe_divide(price_change, abs(ma25_change), 0)
                # 归一化：假设合理范围为[-2, 2]
                # ✅ 修复NaN传播：np.clip会传播NaN，改用max/min
                trend_angle_normalized = max(-1.0, min(trend_angle / 2.0, 1.0)) if np.isfinite(trend_angle) else 0.0
    else:
        trend_angle_normalized = 0.0
    features.append(trend_angle_normalized)  # F05_02
    
    return features


# 导出
__all__ = ['extract_f05_features']
