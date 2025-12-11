"""
特征核心模块 - 通用工具函数

本文件包含所有特征计算使用的通用工具函数
从 feature_extractor.py 提取，保持100%一致
"""
import numpy as np
import math
from scipy.stats import linregress
from typing import Optional

from .config import EPS, DEFAULT_WHEN_ZERO, CFG


# ============================================================================
# 基础工具函数
# ============================================================================

def safe_divide(numerator, denominator, default=DEFAULT_WHEN_ZERO, eps=EPS):
    """安全的除法操作，避免除以零或极小值导致的溢出
    
    ✅ P0修复：检查NaN和Inf，防止数值污染
    """
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError):
        return float(default)
    
    # ✅ P0修复：严格检查有限性（同时检查NaN和Inf）
    if not np.isfinite(n) or not np.isfinite(d):
        return float(default)
    
    if abs(d) < eps:
        return float(default)
    
    res = n / d
    
    # ✅ P0修复：检查结果有限性（防止 large/tiny = Inf）
    if not np.isfinite(res):
        return float(default)
    
    return res


# ============================================================================
# 价格相关工具函数
# ============================================================================

def compute_price_acceleration_correct(closes, idx, short_period=10, long_period=20):
    """改进动量加速度 - 正确的二阶导数计算 + 时间归一化
    
    添加真正的时间归一化，使不同周期参数的结果可比
    边界检查需要long_period+short_period的历史数据
    防止time_step为0导致除零
    """
    # 需要足够数据计算momentum_prev
    if idx < long_period + short_period:
        return 0

    # 计算不同时间点的动量
    momentum_now = safe_divide(closes[idx] - closes[idx-short_period],
                               closes[idx-short_period], default=0, eps=EPS)
    momentum_mid = safe_divide(closes[idx-long_period//2] - closes[idx-long_period//2-short_period],
                               closes[idx-long_period//2-short_period], default=0, eps=EPS)
    momentum_prev = safe_divide(closes[idx-long_period] - closes[idx-long_period-short_period],
                                closes[idx-long_period-short_period], default=0, eps=EPS)

    # 正确的时间归一化（每根K线的加速度变化率）
    # 防止除零
    time_step = max(1, long_period // 2)
    acc_first = (momentum_mid - momentum_prev) / time_step
    acc_second = (momentum_now - momentum_mid) / time_step
    acceleration = acc_second - acc_first  # 每根K线的加速度变化率

    # 归一化因子应与时间步长关联
    # 假设±1%/时间步为合理范围
    normalization_factor = 0.01 * time_step
    acceleration_normalized = max(-1.0, min(1.0, acceleration / normalization_factor))
    
    return acceleration_normalized


def compute_exponential_velocity(current_value, value_array, half_life=5):
    """指数加权平均替代固定窗口
    
    历史数组顺序约定
    
    参数:
        current_value: 当前值（用于计算速度）
        value_array: 历史值数组，**必须按从近到远排序**
                    即 value_array[0] 是最近的历史值（idx-1）
                       value_array[1] 是次近的历史值（idx-2）
                       依此类推
        half_life: 指数衰减半衰期（默认5）
    
    返回:
        float: 当前值相对于历史加权平均的变化速度，范围[-1, 1]
    """
    # ✅ P3修复：添加输入校验，防止NaN/Inf传播
    if not np.isfinite(current_value):
        return 0.0
    
    value_array = np.asarray(value_array)
    if len(value_array) < 2:
        return 0.0
    
    # 清洗NaN/Inf
    value_array_clean = value_array[np.isfinite(value_array)]
    if len(value_array_clean) < 2:
        return 0.0

    # 移除反序，确保最近值权重最大
    # value_array[0] 是最近值，time_indices[0]=0 应获得最大权重 exp(0)=1.0
    time_indices = np.arange(len(value_array_clean))
    weights = np.exp(-time_indices / half_life)
    weights /= weights.sum()

    weighted_avg = np.average(value_array_clean, weights=weights)
    velocity = safe_divide(current_value - weighted_avg, np.abs(weighted_avg) + EPS, 0)

    return max(-1.0, min(1.0, velocity))


# ============================================================================
# ATR相关工具函数
# ============================================================================

def compute_atr_with_momentum(atr_array, idx, short_period=5):
    """ATR融合静态+动态信息"""
    if idx < short_period:
        atr_momentum = 0
    else:
        # 第三个参数应为default=0，而非EPS
        atr_momentum = safe_divide(atr_array[idx] - atr_array[idx-short_period],
                                   atr_array[idx-short_period], default=0, eps=EPS)

    atr_level = atr_array[idx]
    atr_level_norm = safe_divide(atr_level, np.mean(atr_array[max(0, idx-59):idx+1]), 1.0)
    atr_momentum_norm = max(-1.0, min(1.0, atr_momentum))

    combined = 0.3 * atr_level_norm + 0.7 * atr_momentum_norm
    return max(-1.0, min(1.0, combined))


# ============================================================================
# K线形态工具函数
# ============================================================================

def compute_hammer_strength(open_p, close_p, high_p, low_p):
    """锤子线强度计算
    
    ✅ 精确性修复：锤子线应该是阳线（看涨信号）
    阴线的类似形态为吊颈线（看跌信号），应区分处理
    """
    body = abs(close_p - open_p)
    lower_shadow = min(open_p, close_p) - low_p
    upper_shadow = high_p - max(open_p, close_p)
    full_range = high_p - low_p

    if full_range < EPS:
        return 0.0

    hammer_degree = (lower_shadow - upper_shadow) / full_range
    body_ratio = safe_divide(body, full_range, 0)
    hammer_strength = hammer_degree - body_ratio

    # ✅ 精确性修复：锤子线必须是阳线
    if close_p < open_p:
        # 阴线（吊颈线）：降低强度或反转为负值
        hammer_strength *= 0.5  # 降低强度，表示看跌信号

    return max(-1.0, min(1.0, hammer_strength))


def compute_engulfing_strength(open_p, close_p, high_p, low_p, prev_open, prev_close, prev_high, prev_low):
    """吞没形态强度计算
    
    使用标准吞没定义（实体包住实体），而非穿越整个K线
    """
    curr_body = abs(close_p - open_p)
    prev_body = abs(prev_close - prev_open)

    if prev_body < EPS:
        return 0.0

    # 看涨吞没（bullish engulfing）
    # 标准定义：当前阳线实体包住前一阴线实体
    bullish_engulfing = (
        close_p > open_p and           # 当前为阳线
        prev_close < prev_open and     # 前一为阴线
        close_p > prev_open and        # 当前收盘 > 前一开盘
        open_p < prev_close and        # 当前开盘 < 前一收盘
        curr_body > prev_body          # 当前实体 > 前一实体
    )
    
    if bullish_engulfing:
        strength = safe_divide(curr_body - prev_body, prev_body, 0)
        return max(0.0, min(1.0, strength))
    
    # 看跌吞没（bearish engulfing）
    # 标准定义：当前阴线实体包住前一阳线实体
    bearish_engulfing = (
        close_p < open_p and           # 当前为阴线
        prev_close > prev_open and     # 前一为阳线
        close_p < prev_open and        # 当前收盘 < 前一开盘
        open_p > prev_close and        # 当前开盘 > 前一收盘
        curr_body > prev_body          # 当前实体 > 前一实体
    )
    
    if bearish_engulfing:
        strength = safe_divide(curr_body - prev_body, prev_body, 0)
        return -max(0.0, min(1.0, strength))

    return 0.0


def compute_doji_strength(open_p, close_p, high_p, low_p):
    """十字星强度计算
    
    使用配置类的常量，而非硬编码阈值
    """
    body = abs(close_p - open_p)
    true_range = high_p - low_p

    if true_range < EPS:
        return 0.0

    body_ratio = safe_divide(body, true_range, 0)

    # 使用配置类的常量
    if body_ratio < CFG.DOJI_BODY_THRESHOLD_STRONG:
        return 1.0
    elif body_ratio < CFG.DOJI_BODY_THRESHOLD_WEAK:
        return 0.5

    return 0.0


def compute_candle_pattern_combined(open_p, close_p, high_p, low_p, prev_open, prev_close, prev_high, prev_low):
    """K线形态综合得分（合并F01_15/16/17）
    
    合并锤子线、吞没形态、十字星为单一综合特征
    保留最强信号，避免特征稀疏性
    
    参数:
        open_p, close_p, high_p, low_p: 当前K线OHLC
        prev_open, prev_close, prev_high, prev_low: 前一根K线OHLC
    
    返回:
        float: [-1, 1] K线形态综合得分
               正值=看涨形态，负值=看跌形态，0=无明显形态
    """
    # 计算三种形态强度
    hammer = compute_hammer_strength(open_p, close_p, high_p, low_p)
    engulfing = compute_engulfing_strength(open_p, close_p, high_p, low_p,
                                          prev_open, prev_close, prev_high, prev_low)
    doji = compute_doji_strength(open_p, close_p, high_p, low_p)
    
    # 保留绝对值最大的信号（最强形态）
    signals = [hammer, engulfing, doji * 0.5]  # 十字星权重降低（反转信号较弱）
    abs_signals = [abs(s) for s in signals]
    max_idx = abs_signals.index(max(abs_signals))
    
    return signals[max_idx]


def compute_close_position(open_p, close_p, high_p, low_p):
    """K线收盘位置"""
    full_range = high_p - low_p

    if full_range < EPS:
        return 0.5 if close_p >= open_p else -0.5

    position_ratio = safe_divide(close_p - low_p, full_range, 0.5)
    close_position = (position_ratio - 0.5) * 2.0

    open_position_ratio = safe_divide(open_p - low_p, full_range, 0.5)
    open_center_distance = abs(open_position_ratio - 0.5)
    weight = 1.0 - open_center_distance * 0.3

    return max(-1.0, min(1.0, close_position * weight))


# ============================================================================
# 成交量相关工具函数
# ============================================================================

def compute_volume_percentile(current_volume, volume_history):
    """成交量绝对百分位
    
    添加NaN/Inf处理
    """
    if len(volume_history) < 5:
        return 0.5

    # 清洗NaN/Inf
    vh = np.array(volume_history)
    vh = vh[np.isfinite(vh)]
    
    if len(vh) < 5:
        return 0.5

    # 使用<=而非<，符合标准百分位定义
    rank = np.sum(vh <= current_volume) / len(vh)

    # 修复：移除对极高成交量的向下调整逻辑
    # 原逻辑会把95%百分位压低到[0.5,1.0]区间，失去区分度
    # 现在直接返回原始百分位，保留特征的判别力
    return max(0.0, min(1.0, rank))


def compute_price_volume_correlation(price_changes, volume_changes, window=5):
    """价量相关性强度
    
    ✅ 审计修复：改用Pearson相关系数，基于窗口数据计算趋势相关性
    
    参数:
        price_changes: 价格变化率数组（最近window根K线）
        volume_changes: 成交量变化率数组（最近window根K线）
        window: 计算窗口大小（默认5）
    
    返回:
        float: [-1, 1] 范围的Pearson相关系数
            +1 = 完美正相关（价涨量也涨）
             0 = 无相关
            -1 = 完美负相关（价涨量却缩）
    """
    # 检查数据充分性
    if len(price_changes) < window or len(volume_changes) < window:
        return 0.0
    
    # 提取最近window个值
    recent_prices = np.array(price_changes[-window:])
    recent_volumes = np.array(volume_changes[-window:])
    
    # 清洗NaN/Inf
    valid_mask = np.isfinite(recent_prices) & np.isfinite(recent_volumes)
    if np.sum(valid_mask) < 3:  # 至少需3个有效点
        return 0.0
    
    recent_prices = recent_prices[valid_mask]
    recent_volumes = recent_volumes[valid_mask]
    
    # 计算Pearson相关系数
    try:
        # 检查是否所有值都相同（无变化）
        if np.std(recent_prices) < EPS or np.std(recent_volumes) < EPS:
            return 0.0
        
        correlation_matrix = np.corrcoef(recent_prices, recent_volumes)
        correlation = correlation_matrix[0, 1]
        
        # ✅ P2修复：检查有限性（同时检查NaN和Inf）
        if not np.isfinite(correlation):
            return 0.0
        
        return max(-1.0, min(1.0, correlation))
    except Exception:
        return 0.0


def compute_capital_persistence(daily_volumes_array, ma60_vol_baseline, window=5):
    """资金持续关注度 - 最近N根中有多少根持续放量
    
    修正参数语义，应比较日成交量 vs MA60基准
    
    参数:
        daily_volumes_array: 日成交量数组（最近window根）
        ma60_vol_baseline: MA60成交量基准值
        window: 窗口大小（默认5）
    """
    if len(daily_volumes_array) < window:
        return 0.5

    recent_volumes = np.array(daily_volumes_array[-window:])
    above_baseline = np.sum(recent_volumes > ma60_vol_baseline * 1.2)  # 日成交量 vs MA60
    persistence = above_baseline / window

    if window >= 2:
        vol_trend = recent_volumes[-1] - recent_volumes[0]
        vol_trend_normalized = safe_divide(vol_trend, ma60_vol_baseline, 0)
        vol_trend_signal = max(-1.0, min(1.0, vol_trend_normalized))
        final_score = 0.6 * persistence + 0.4 * (vol_trend_signal + 1.0) / 2.0
    else:
        final_score = persistence

    return max(0.0, min(1.0, final_score))


def compute_volume_extreme_event(current_vol, volume_history):
    """成交量极值事件
    
    ✅ 审计修复：最小样本数改为30（原来10太少，统计不稳定）
    ✅ P2修复：MIN_REQUIRED动态计算，与窗口大小成比例（至少保证80%）
    添加NaN/Inf处理
    """
    # ✅ P2修复：动态计算最小样本数（与窗口大小成比例）
    # 窗口=60 → MIN_REQUIRED=48 (80%)
    # 窗口=59 → MIN_REQUIRED=47 (80%)
    MIN_REQUIRED = max(int(len(volume_history) * 0.8), 40)  # 至少保证80%，最低40个
    
    if len(volume_history) < MIN_REQUIRED:
        return 0

    # 清洗NaN/Inf
    vh = np.array(volume_history)
    vh = vh[np.isfinite(vh)]
    
    if len(vh) < MIN_REQUIRED:  # ✅ 审计修复：清洗后也要检查
        return 0

    p10 = np.percentile(vh, 10)
    p25 = np.percentile(vh, 25)
    p75 = np.percentile(vh, 75)
    p90 = np.percentile(vh, 90)

    if current_vol > p90:
        # ✅ P0修复：max(EPS, x)在x=NaN时返回NaN，改用显式检查
        # ✅ 修复NaN传播：np.max对已过滤数组安全，但加上检查更稳健
        vh_max = np.max(vh) if len(vh) > 0 and np.all(np.isfinite(vh)) else current_vol
        range_high = vh_max - p90
        den_high = range_high if np.isfinite(range_high) and range_high > EPS else EPS
        extreme_level = safe_divide(current_vol - p90, den_high, 0)
        return min(1.0, extreme_level)
    elif current_vol > p75:
        return 0.3
    elif current_vol < p10:
        # ✅ P0修复：max(EPS, x)在x=NaN时返回NaN，改用显式检查
        # ✅ 修复NaN传播：np.min对已过滤数组安全，但加上检查更稳健
        vh_min = np.min(vh) if len(vh) > 0 and np.all(np.isfinite(vh)) else current_vol
        range_low = p10 - vh_min
        den_low = range_low if np.isfinite(range_low) and range_low > EPS else EPS
        extreme_level = safe_divide(p10 - current_vol, den_low, 0)
        return -min(1.0, extreme_level)
    elif current_vol < p25:
        return -0.3
    else:
        return 0


# ============================================================================
# 均线斜率计算（线性回归版）
# ============================================================================

def compute_ma25_slope_atr_normalized_linregress(ma25_array, atr_array, idx, period=25):
    """MA25斜率线性回归计算，同时返回R²
    
    使用25个点做线性回归，而非只用两个点
    优点：
        - 更准确的趋势方向（能过滤异常点）
        - R²值，反映趋势稳定性（能区分稳定上涨 vs 震荡上涨）

    参数:
        ma25_array: MA25数组
        atr_array: ATR数组
        idx: 当前索引
        period: 计算斜率的周期（默认25）

    返回:
        tuple: (slope_normalized, r_squared)
            - slope_normalized: [-1, 1] 范围的ATR归一化斜率
            - r_squared: [0, 1] 范围的拟合优度（1=完美直线，0=完全混乱）
    """
    if idx < period:
        return 0.0, 0.0
    
    # ✅ P0修复：检查有限性（同时检查NaN和Inf）
    ma25_window = ma25_array[idx-period+1:idx+1]  # 取25个点
    if not np.all(np.isfinite(ma25_window)) or not np.isfinite(atr_array[idx]):
        return 0.0, 0.0

    # 线性回归：y = slope * x + intercept
    x = np.arange(period)  # [0, 1, 2, ..., 24]
    y = ma25_window
    
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2  # 拟合优度
    except Exception:
        return 0.0, 0.0

    # ✅ 精确性修复：ATR极小值保护，避免极低波动时的数值不稳定
    # ✅ 审计优化：使用相对阈值而非绝对阈值，适应不同价格区间的股票
    atr_val = atr_array[idx]
    # 相对阈值：MA25价格的0.01%
    current_price = ma25_array[idx-period+1:idx+1][-1] if len(ma25_array[idx-period+1:idx+1]) > 0 else 1.0
    min_atr_threshold = current_price * 0.0001
    if atr_val < min_atr_threshold:
        return 0.0, r_squared

    # slope是每个K线的变化量，乘以period得到总变化
    slope_atr_ratio = safe_divide(slope * period, atr_val, 0)
    slope_atr_ratio = max(-1.0, min(1.0, slope_atr_ratio))

    return slope_atr_ratio, r_squared


def compute_ma5_slope_atr_normalized_linregress(ma5_array, atr_array, idx, period=5):
    """MA5斜率（线性回归版）同时返回R²
    
    用最近5根K线做线性回归，判断MA5趋势方向和稳定性
    配合F01_06（MA25斜率）可以判断：
        - 两线是否都向上
        - 是粘在一起向上，还是分开向上

    参数:
        ma5_array: MA5数组
        atr_array: ATR数组
        idx: 当前索引
        period: 计算斜率的周期（默认5）

    返回:
        tuple: (slope_normalized, r_squared)
            - slope_normalized: [-1, 1] 范围的ATR归一化斜率
            - r_squared: [0, 1] 范围的拟合优度（1=完美直线，0=混乱）
    """
    if idx < period:
        return 0.0, 0.0
    
    # ✅ P0修复：检查有限性（同时检查NaN和Inf）
    ma5_window = ma5_array[idx-period+1:idx+1]  # 取5个点
    if not np.all(np.isfinite(ma5_window)) or not np.isfinite(atr_array[idx]):
        return 0.0, 0.0

    # 线性回归：y = slope * x + intercept
    x = np.arange(period)  # [0, 1, 2, 3, 4]
    y = ma5_window
    
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2  # 拟合优度
    except Exception:
        return 0.0, 0.0

    # ✅ 精确性修复：ATR极小值保护，避免极低波动时的数值不稳定
    # ✅ 审计优化：使用相对阈值而非绝对阈值，适应不同价格区间的股票
    atr_val = atr_array[idx]
    # 相对阈值：MA5价格的0.01%
    current_price = ma5_array[idx-period+1:idx+1][-1] if len(ma5_array[idx-period+1:idx+1]) > 0 else 1.0
    min_atr_threshold = current_price * 0.0001
    if atr_val < min_atr_threshold:
        return 0.0, r_squared

    # slope是每个K线的变化量，乘以period得到总变化
    slope_atr_ratio = safe_divide(slope * period, atr_val, 0)
    slope_atr_ratio = max(-1.0, min(1.0, slope_atr_ratio))

    return slope_atr_ratio, r_squared


def compute_volume_ma_slope_linregress(vol_ma_array, idx, period, vol_ma60_array):
    """量线MA斜率（线性回归版） + 稳定性（R²）
    
    用指定个K线做线性回归，判断量线趋势方向和稳定性
    用MA60量作为基准进行归一化（类似价格特征用ATR）

    参数:
        vol_ma_array: 量线MA数组（MA5或MA60）
        idx: 当前索引
        period: 计算斜率的周期（5或60）
        vol_ma60_array: MA60量数组（用于归一化）

    返回:
        tuple: (slope_normalized, r_squared)
            - slope_normalized: [-1, 1] 范围的归一化斜率
            - r_squared: [0, 1] 范围的拟合优度（1=完美直线，0=混乱）
    """
    if idx < period:
        return 0.0, 0.0
    
    # ✅ P0修复：检查有限性（同时检查NaN和Inf）
    vol_ma_window = vol_ma_array[idx-period+1:idx+1]  # 取period个点
    if not np.all(np.isfinite(vol_ma_window)) or not np.isfinite(vol_ma60_array[idx]):
        return 0.0, 0.0

    # 线性回归：y = slope * x + intercept
    x = np.arange(period)
    y = vol_ma_window
    
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2  # 拟合优度
    except Exception:
        return 0.0, 0.0

    # ✅ 精确性修复：MA60量极小值保护，避免极低成交量时的数值不稳定
    vol_ma60_val = vol_ma60_array[idx]
    if vol_ma60_val < EPS * 100:  # MA60量过小，斜率不可靠
        return 0.0, r_squared

    # slope是每个K线的变化量，乘以period得到总变化
    slope_vol_ratio = safe_divide(slope * period, vol_ma60_val, 0)
    slope_vol_ratio = max(-1.0, min(1.0, slope_vol_ratio))

    return slope_vol_ratio, r_squared


def compute_volume_ratio_metrics(
    ma5_vol: float,
    ma60_vol: float,
    ma5_vol_array = None,
    ma60_vol_array = None,
    idx = None,
    lookback_period: int = 20
) -> dict:
    """计算5日成交量与60日成交量的占比和高度关系"""
    
    result = {
        'current_above_ratio': 1.0 if ma5_vol > ma60_vol else 0.0,
        'above_ratio': 0.5,
        'above_duration': 0,
        'below_duration': 0,
        'volume_momentum': 0.0,
        'is_volume_expanding': False,
        'is_volume_contracting': False,
    }
    
    if ma5_vol_array is None or ma60_vol_array is None or idx is None:
        if ma60_vol > 1e-8:
            result['volume_momentum'] = safe_divide(ma5_vol - ma60_vol, ma60_vol, 0)
            result['volume_momentum'] = max(-1.0, min(1.0, result['volume_momentum']))
        return result
    
    if idx < 1:
        # ✅ NaN防护：返回安全默认值而非NaN，避免NaN传播
        result['above_ratio'] = 0.5
        result['volume_momentum'] = 0.0
        return result
    
    try:
        lookback = min(idx + 1, lookback_period)
        start_idx = max(0, idx - lookback + 1)
        
        ma5_window = ma5_vol_array[start_idx:idx+1]
        ma60_window = ma60_vol_array[start_idx:idx+1]
        
        if (len(ma5_window) < 2 or len(ma60_window) < 2 or
            not np.all(np.isfinite(ma5_window)) or
            not np.all(np.isfinite(ma60_window))):
            return result
        
        above_mask = ma5_window > ma60_window
        result['above_ratio'] = np.mean(above_mask)

        # ✅ 向量化：计算从后往前连续上升/下降的天数
        # 先翻转数组，然后找第一个False的位置
        above_mask_reversed = above_mask[::-1]
        below_mask_reversed = ~above_mask_reversed

        # 计算连续True的长度（从末尾开始）
        above_duration = 0
        for val in above_mask_reversed:
            if val:
                above_duration += 1
            else:
                break

        below_duration = 0
        for val in below_mask_reversed:
            if val:
                below_duration += 1
            else:
                break

        result['above_duration'] = above_duration
        result['below_duration'] = below_duration
        
        current_diff = ma5_vol - ma60_vol
        
        if ma60_vol > 1e-8:
            diff_ratio = current_diff / ma60_vol
            result['volume_momentum'] = np.tanh(diff_ratio * 3)
        
        if (result['above_ratio'] > 0.6 and
            above_duration >= 3 and
            result['volume_momentum'] > 0.1):
            result['is_volume_expanding'] = True
        
        if (result['above_ratio'] < 0.4 and
            below_duration >= 3 and
            result['volume_momentum'] < -0.1):
            result['is_volume_contracting'] = True
        
    except Exception:
        # ✅ NaN防护：异常时返回安全默认值而非NaN
        result['above_ratio'] = 0.5
        result['volume_momentum'] = 0.0
    
    return result


def calculate_atr_recent_60bars(
    high_array: np.ndarray,
    low_array: np.ndarray,
    close_array: np.ndarray,
    idx: int,
    period: int = 60
) -> float:
    """计算最近N根K线的ATR值（真实波动幅度均值）"""
    EPS = 1e-8
    
    if idx < period - 1:
        return np.nan
    
    start_idx = max(0, idx - period + 1)
    end_idx = idx + 1
    
    high_window = high_array[start_idx:end_idx]
    low_window = low_array[start_idx:end_idx]
    close_window = close_array[start_idx:end_idx]
    
    if (len(high_window) < period or len(low_window) < period or len(close_window) < period):
        return np.nan
    
    if not (np.all(np.isfinite(high_window)) and np.all(np.isfinite(low_window)) and
            np.all(np.isfinite(close_window))):
        return np.nan
    
    try:
        tr_array = np.zeros(period)
        
        tr_array[0] = high_window[0] - low_window[0]
        
        for i in range(1, period):
            hl_range = high_window[i] - low_window[i]
            h_pc_range = abs(high_window[i] - close_window[i - 1])
            l_pc_range = abs(low_window[i] - close_window[i - 1])
            
            tr_array[i] = max(hl_range, h_pc_range, l_pc_range)
        
        if not np.all(np.isfinite(tr_array)) or np.all(tr_array < EPS):
            return np.nan
        
        atr_value = np.mean(tr_array)
        
        if not np.isfinite(atr_value) or atr_value < 0:
            return np.nan
        
        return float(atr_value)
        
    except Exception:
        return np.nan


# ============================================================================
# 导出所有工具函数
# ============================================================================
__all__ = [
    'safe_divide',
    'compute_price_acceleration_correct',
    'compute_exponential_velocity',
    'compute_atr_with_momentum',
    'compute_hammer_strength',
    'compute_engulfing_strength',
    'compute_doji_strength',
    'compute_candle_pattern_combined',
    'compute_close_position',
    'compute_volume_percentile',
    'compute_price_volume_correlation',
    'compute_capital_persistence',
    'compute_volume_extreme_event',
    'compute_ma25_slope_atr_normalized_linregress',
    'compute_ma5_slope_atr_normalized_linregress',
    'compute_volume_ma_slope_linregress',
    'compute_volume_ratio_metrics',
    'calculate_atr_recent_60bars',
    # 新增：基础计算函数
    'rolling_mean_aligned',
    'calculate_ema',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_rsi_standard',
    'calculate_kdj_standard',
]


# ============================================================================
# 基础计算函数（用于特征提取前的指标计算）
# ============================================================================

def rolling_mean_aligned(arr, window):
    """
    计算与原数组长度相同的移动平均（前 window-1 个值为 NaN）
    ✅ 支持NaN处理：如果窗口内所有值都是NaN，结果为NaN；否则忽略NaN计算平均值
    """
    res = np.full(len(arr), np.nan)
    if len(arr) >= window:
        # ✅ 使用 np.nanmean 代替 np.convolve，自动忽略 NaN 值
        for i in range(window - 1, len(arr)):
            window_values = arr[i - window + 1:i + 1]
            # 如果窗口内所有值都是NaN，结果为NaN
            if np.all(np.isnan(window_values)):
                res[i] = np.nan
            else:
                # 否则忽略NaN计算平均值
                res[i] = np.nanmean(window_values)
    return res


def calculate_ema(data, period):
    """
    计算指数移动平均线（EMA）
    
    ✅ 关键修复：不生成NaN，使用传统递推方法以兼容现有数据
    
    参数：
        data: 价格数据数组
        period: EMA周期
    
    返回：
        EMA数组
    """
    ema = np.zeros(len(data))
    alpha = 2.0 / (period + 1)
    
    # ✅ 关键修复：使用传统方法，不生成NaN
    if len(data) >= period:
        # 使用SMA作为EMA初始值
        ema[period-1] = np.mean(data[:period])
        # 前period-1个值使用简单递推
        for i in range(period-1):
            if i == 0:
                ema[i] = data[i]
            else:
                ema[i] = ema[i-1] * (1 - alpha) + data[i] * alpha
        # 从第period个值开始使用EMA公式
        for i in range(period, len(data)):
            ema[i] = ema[i-1] * (1 - alpha) + data[i] * alpha
    else:
        # 数据不足period，使用传统方法
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = ema[i-1] * (1 - alpha) + data[i] * alpha
    
    return ema


def calculate_atr(highs, lows, closes, period=14):
    """
    计算平均真实波动幅度（ATR）- 标准Wilder实现
    
    ✅ 审计修复：前period-1个值为NaN，第period个值使用SMA初始化
    
    参数:
        highs: 最高价数据数组
        lows: 最低价数据数组
        closes: 收盘价数据数组
        period: ATR周期（默认14）
    
    返回:
        ATR数组（前period-1个值为NaN，从第period个值开始有效）
    """
    atr = np.full(len(closes), np.nan)  # ✅ 初始化为NaN而非0
    
    if len(closes) < period:
        return atr
    
    # ✅ 向量化计算真实波幅（True Range）
    hl_range = highs - lows

    # 对于i>=1，计算高-前收和低-前收的绝对值
    h_prev_close = np.abs(highs[1:] - closes[:-1])
    l_prev_close = np.abs(lows[1:] - closes[:-1])

    # 创建向量化的true_ranges数组
    true_ranges = np.zeros(len(closes))
    true_ranges[0] = hl_range[0]

    # ✅ NaN防护：先清洗NaN，再计算maximum
    h_prev_close = np.where(np.isfinite(h_prev_close), h_prev_close, 0.0)
    l_prev_close = np.where(np.isfinite(l_prev_close), l_prev_close, 0.0)
    hl_range_1 = np.where(np.isfinite(hl_range[1:]), hl_range[1:], 0.0)

    max_h_prev = np.maximum(h_prev_close, l_prev_close)
    true_ranges[1:] = np.maximum(hl_range_1, max_h_prev)
    
    # ✅ 审计修复：第period-1个位置使用SMA作为ATR初始值（标准Wilder方法）
    # ✅ NaN防护：清洗true_ranges中的NaN后计算均值
    true_ranges_clean = true_ranges[:period]
    true_ranges_clean = np.where(np.isfinite(true_ranges_clean), true_ranges_clean, 0.0)
    atr_init = np.mean(true_ranges_clean)
    if np.isfinite(atr_init) and atr_init > 0:
        atr[period-1] = atr_init
    else:
        atr[period-1] = np.nan
    
    # 从第period个值开始使用Wilder平滑法
    for i in range(period, len(closes)):
        atr[i] = (atr[i-1] * (period - 1) + true_ranges[i]) / period
    
    return atr


def calculate_bollinger_bands(data, period=20, num_std=2):
    """
    计算布林带（上轨、中轨、下轨）
    
    ✅ 精确性修复：数据不足时输出NaN，符合金融标准
    
    参数:
        data: 价格数据数组
        period: 布林带周期（默认20）
        num_std: 标准差倍数（默认2）
    
    返回:
        (upper_band, middle_band, lower_band) 元组
    """
    upper_band = np.full(len(data), np.nan)
    middle_band = np.full(len(data), np.nan)
    lower_band = np.full(len(data), np.nan)
    
    # ✅ 精确性修复：前period-1个值保持为NaN，只计算完整周期的布林带
    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        middle = np.mean(window)
        std = np.std(window, ddof=1)  # ✅ 使用样本标准差（金融标准）
        
        middle_band[i] = middle
        upper_band[i] = middle + num_std * std
        lower_band[i] = middle - num_std * std
    
    return upper_band, middle_band, lower_band


def calculate_rsi_standard(closes, period=14):
    """
    计算相对强弱指标（RSI）- 标准Wilder实现
    
    参数:
        closes: 收盘价数据数组
        period: RSI周期（默认14）
    
    返回:
        RSI数组 (0-100)
    """
    rsi = np.full(len(closes), 50.0)  # 默认值50
    
    if len(closes) < period + 1:
        return rsi
    
    # ✅ 向量化计算价格变化、收益和损失
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # ✅ 向量化：第一个RSI使用SMA（前period个值的平均）
    avg_gain = np.mean(gains[:period]) if len(gains) >= period else 0.0
    avg_loss = np.mean(losses[:period]) if len(losses) >= period else 0.0
    
    if avg_loss < EPS:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # 后续使用Wilder平滑法
    for i in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        
        if avg_loss < EPS:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_kdj_standard(closes, highs, lows, n=9, m1=3, m2=3):
    """
    计算KDJ指标 - 标准实现
    
    参数:
        closes: 收盘价数据数组
        highs: 最高价数据数组
        lows: 最低价数据数组
        n: RSV周期（默认9）
        m1: K值平滑参数（默认3）
        m2: D值平滑参数（默认3）
    
    返回:
        (k, d, j) 元组
    """
    k = np.full(len(closes), 50.0)
    d = np.full(len(closes), 50.0)
    j = np.full(len(closes), 50.0)
    
    if len(closes) < n:
        return k, d, j
    
    # ✅ 向量化计算窗口最高价和最低价
    # 对于每个i从n-1开始，计算[i-n+1:i+1]窗口内的最高和最低价
    # 使用 np.lib.stride_tricks 或直接循环计算窗口（KDJ需要递推，难以完全向量化）
    for i in range(n - 1, len(closes)):
        # 向量化：计算窗口最高最低价
        window_highs = highs[max(0, i-n+1):i+1]
        window_lows = lows[max(0, i-n+1):i+1]
        # ✅ 修复NaN传播：过滤NaN后再计算max/min
        window_highs_clean = window_highs[np.isfinite(window_highs)]
        window_lows_clean = window_lows[np.isfinite(window_lows)]
        window_high = np.max(window_highs_clean) if len(window_highs_clean) > 0 else 0
        window_low = np.min(window_lows_clean) if len(window_lows_clean) > 0 else 0

        if window_high - window_low < EPS:
            rsv = 50.0
        else:
            rsv = 100.0 * (closes[i] - window_low) / (window_high - window_low)

        # K倽 = (1/m1) * RSV + ((m1-1)/m1) * K倽-1
        k[i] = (rsv / m1) + (k[i-1] * (m1 - 1) / m1)
        # D值 = (1/m2) * K值 + ((m2-1)/m2) * D值-1
        d[i] = (k[i] / m2) + (d[i-1] * (m2 - 1) / m2)
        # J值 = 3*K值 - 2*D值
        j[i] = 3 * k[i] - 2 * d[i]
    
    return k, d, j
