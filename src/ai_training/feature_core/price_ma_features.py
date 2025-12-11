"""
价格均线特征组（F01_01~F01_13）

本模块包含所有与价格均线相关的13个特征计算
从 feature_extractor.py 提取，保持100%一致
"""
import numpy as np
from .config import EPS
from .utils import (
    safe_divide,
    compute_ma25_slope_atr_normalized_linregress,
    compute_ma5_slope_atr_normalized_linregress,
    compute_exponential_velocity,
    compute_candle_pattern_combined,
)


def extract_f01_features(
    idx, close, open_price, high, low,
    opens, closes, highs, lows,
    ma5_prices, ma25_prices, atr
):
    """
    提取价格均线特征组（F01_01~F01_13）
    
    参数:
        idx: 当前K线索引
        close: 当前收盘价
        open_price: 当前开盘价
        high: 当前最高价
        low: 当前最低价
        opens: 开盘价数组
        closes: 收盘价数组
        highs: 最高价数组
        lows: 最低价数组
        ma5_prices: MA5均线数组
        ma25_prices: MA25均线数组
        atr: ATR数组
    
    返回:
        list: 包含13个特征值的列表 [F01_01, F01_02, ..., F01_13]
    """
    features = []
    
    # 当前均线值
    ma5 = ma5_prices[idx]
    ma25 = ma25_prices[idx]
    
    # === F01_01: MA5价格归一化 ===
    # ✅ Bug#11修复：调整MA归一化范围，提高区分度
    # A股实际情况：MA5通常在close的±5%范围内
    ma5_norm = safe_divide(ma5, close, 1.0)
    # 将[0.95, 1.05]映射到[-1, 1]，超出范围裁剪
    ma5_norm = max(-1.0, min(1.0, (ma5_norm - 1.0) * 20))  # ±5% -> ±1.0
    features.append(ma5_norm)  # F01_01
    
    # === F01_02: MA25价格归一化 [P0-2560战法核心] ===
    ma25_norm = safe_divide(ma25, close, 1.0)
    # MA25波动更大，使用±10%映射
    ma25_norm = max(-1.0, min(1.0, (ma25_norm - 1.0) * 10))  # ±10% -> ±1.0
    features.append(ma25_norm)  # F01_02
    
    # === F01_03: MA5>MA25（金叉状态） ===
    features.append(1.0 if ma5 > ma25 else 0.0)  # F01_03
    
    # === F01_04/05: MA25趋势斜率+稳定性（线性回归版） [P0-2560战法核心] ===
    # 改进：使用25个点做线性回归，同时得到斜率和稳定性两个特征
    if idx >= 25:
        try:
            ma25_slope_atr, ma25_r_squared = compute_ma25_slope_atr_normalized_linregress(
                ma25_prices, atr, idx, period=25
            )
            # ✅ NaN防护：检查返回值的有效性
            if not np.isfinite(ma25_slope_atr):
                ma25_slope_atr = 0.0
            if not np.isfinite(ma25_r_squared):
                ma25_r_squared = 0.0
        except:
            ma25_slope_atr = 0.0
            ma25_r_squared = 0.0
    else:
        ma25_slope_atr = 0.0
        ma25_r_squared = 0.0
    features.append(ma25_slope_atr)  # F01_04: MA25趋势斜率（线性回归版）
    features.append(ma25_r_squared)  # F01_05: MA25趋势稳定性（R²）
    
    # === F01_06/07: MA5趋势斜率+稳定性（线性回归版） ===
    # 用最近5根K线做线性回归，判断MA5趋势方向
    # 配合F01_04可以判断：两线是否都向上，是粘在一起还是分开向上
    if idx >= 5:
        try:
            ma5_slope_atr, ma5_r_squared = compute_ma5_slope_atr_normalized_linregress(
                ma5_prices, atr, idx, period=5
            )
            # ✅ NaN防护：检查返回值的有效性
            if not np.isfinite(ma5_slope_atr):
                ma5_slope_atr = 0.0
            if not np.isfinite(ma5_r_squared):
                ma5_r_squared = 0.0
        except:
            ma5_slope_atr = 0.0
            ma5_r_squared = 0.0
    else:
        ma5_slope_atr = 0.0
        ma5_r_squared = 0.0
    features.append(ma5_slope_atr)  # F01_06: MA5趋势斜率（线性回归版）
    features.append(ma5_r_squared)  # F01_07: MA5趋势稳定性（R²）
    
    # === F01_08: K线方向强度 ===
    # 修复：保留连续性，使用K线实体相对于全幅的比例
    body_size = abs(close - open_price)
    full_range = high - low
    if full_range > EPS:
        k_line_strength = safe_divide(body_size, full_range, 0)
        # 保留方向：涨为正，跌为负
        if close < open_price:
            k_line_strength = -k_line_strength
    else:
        k_line_strength = 0
    features.append(k_line_strength)  # F01_08
    
    # === F01_09: MA5-MA25粘合度 ★P0纠缠态 ===
    # ✅ P0修复：语义错误 - "粘合度"应该是1.0表示完全粘合，0表示完全分离
    # ✅ P3修复：安全分母构造，防止ma25=NaN时max返回NaN
    ma25_safe = ma25 if np.isfinite(ma25) and ma25 > EPS else EPS
    distance = safe_divide(abs(ma5 - ma25), ma25_safe, 0)
    distance = min(distance, 1.0)  # 上限1.0（100%分离）
    ma_cohesion = 1.0 - distance  # 反转：1-距离 = 粘合度
    features.append(ma_cohesion)  # F01_09
    
    # === F01_10: MA5-MA25发散速度 ★P0纠缠态 ===
    # (改进：指数加权平均替代固定5根窗口)
    # ✅ Bug#17修复：历史数组不应包含当前值
    # ✅ P0修复：历史数组计算也必须使用粘合度（1.0 - distance），与F01_09保持一致
    if idx >= 10:
        # ✅ 向量化计算：提取历史数据 [idx-10, idx-1]
        look_back_start = idx - 10
        look_back_end = idx

        hist_indices = np.arange(look_back_start, look_back_end)
        hist_ma5 = ma5_prices[hist_indices]
        hist_ma25 = ma25_prices[hist_indices]

        # 向量化检查有效性
        valid_mask = np.isfinite(hist_ma5) & np.isfinite(hist_ma25)

        if np.sum(valid_mask) > 0:
            # 提取有效数据
            valid_ma5 = hist_ma5[valid_mask]
            valid_ma25 = hist_ma25[valid_mask]

            # 向量化计算：安全分母构造
            ma25_safe = np.where(valid_ma25 > EPS, valid_ma25, EPS)

            # 向量化计算距离和粘合度
            distance = np.abs(valid_ma5 - valid_ma25) / ma25_safe
            # ✅ 修复NaN传播：对向量操作，先过滤NaN再clip
            distance = np.where(np.isfinite(distance), np.minimum(distance, 1.0), 1.0)  # NaN视为完全分离
            ma_cohesion_history = 1.0 - distance  # ✅ 修复：使用粘合度，与F01_09一致

            if len(ma_cohesion_history) > 1:
                try:
                    ma_divergence_speed = compute_exponential_velocity(
                        ma_cohesion, ma_cohesion_history, half_life=5
                    )
                    # ✅ NaN防护：检查返回值的有效性
                    if not np.isfinite(ma_divergence_speed):
                        ma_divergence_speed = 0.0
                except:
                    ma_divergence_speed = 0.0
            else:
                ma_divergence_speed = 0
        else:
            ma_divergence_speed = 0
    else:
        ma_divergence_speed = 0
    features.append(ma_divergence_speed)  # F01_10
    
    # === F01_11: K线形态综合得分 ===
    # 合并：锤子线 + 吞没形态 + 十字星 → 单一综合特征
    if idx > 0:
        try:
            pattern_combined = compute_candle_pattern_combined(
                open_price, close, high, low,
                opens[idx-1], closes[idx-1], highs[idx-1], lows[idx-1]
            )
            # ✅ NaN防护：检查返回值的有效性
            if not np.isfinite(pattern_combined):
                pattern_combined = 0.0
        except:
            pattern_combined = 0.0
    else:
        pattern_combined = 0.0
    features.append(pattern_combined)  # F01_11
    
    # === F01_12: K线实体穿越MA5检测 ===
    # 判断整个K线实体是否穿越均线，而非仅收盘价
    if idx > 0:
        prev_open = opens[idx-1]
        prev_close = closes[idx-1]
        prev_ma5 = ma5_prices[idx-1]
        
        # 前一根K线的实体上下沿
        prev_body_high = max(prev_open, prev_close)
        prev_body_low = min(prev_open, prev_close)
        
        # 当前K线的实体上下沿
        curr_body_high = max(open_price, close)
        curr_body_low = min(open_price, close)
        
        # 判断穿越：
        # 向上穿越：前一根实体全在MA5下方 AND 当前实体全在MA5上方
        # 或者：当前K线实体跨越MA5（开盘在下，收盘在上）
        if (prev_body_high < prev_ma5) and (curr_body_low > ma5):
            # 前一根完全在下，当前完全在上 → 向上穿越
            cross_ma5 = 1.0
        elif (open_price < ma5) and (close > ma5):
            # 当前K线开在MA5下，收在MA5上 → 向上穿越
            cross_ma5 = 1.0
        elif (prev_body_low > prev_ma5) and (curr_body_high < ma5):
            # 前一根完全在上，当前完全在下 → 向下穿越
            cross_ma5 = -1.0
        elif (open_price > ma5) and (close < ma5):
            # 当前K线开在MA5上，收在MA5下 → 向下穿越
            cross_ma5 = -1.0
        else:
            cross_ma5 = 0.0  # 没有穿越
    else:
        cross_ma5 = 0.0
    features.append(cross_ma5)  # F01_12
    
    # === F01_13: K线实体穿越MA25检测 ===
    if idx > 0:
        prev_open = opens[idx-1]
        prev_close = closes[idx-1]
        prev_ma25 = ma25_prices[idx-1]
        
        # 前一根K线的实体上下沿
        prev_body_high = max(prev_open, prev_close)
        prev_body_low = min(prev_open, prev_close)
        
        # 当前K线的实体上下沿
        curr_body_high = max(open_price, close)
        curr_body_low = min(open_price, close)
        
        # 判断穿越
        if (prev_body_high < prev_ma25) and (curr_body_low > ma25):
            cross_ma25 = 1.0  # 向上穿越
        elif (open_price < ma25) and (close > ma25):
            cross_ma25 = 1.0  # 当前K线向上穿越
        elif (prev_body_low > prev_ma25) and (curr_body_high < ma25):
            cross_ma25 = -1.0  # 向下穿越
        elif (open_price > ma25) and (close < ma25):
            cross_ma25 = -1.0  # 当前K线向下穿越
        else:
            cross_ma25 = 0.0  # 没有穿越
    else:
        cross_ma25 = 0.0
    features.append(cross_ma25)  # F01_13
    
    return features


# 导出
__all__ = ['extract_f01_features']
