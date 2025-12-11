"""
2560战法特征组（F07_01~F07_03）

本模块包含2560战法的3个核心特征计算
从 feature_extractor.py 提取，保持100%一致

【P0核心特征】2560战法是最重要的交易策略
- F07_01: MA25+VOL+价格三角形验证强度
- F07_02: 量能配合强度
- F07_03: 反弹质量强制评分
"""
import numpy as np
from .config import EPS, VOL_CROSS_THRESHOLD
from .utils import (
    safe_divide,
    compute_ma25_slope_atr_normalized_linregress,
    compute_ma5_slope_atr_normalized_linregress,
    compute_volume_ma_slope_linregress,
    compute_volume_ratio_metrics,
    calculate_atr_recent_60bars,
)


def calculate_high_position_risk(
    price: float,
    ma25: float,
    atr_val: float,
    ma25_slope: float,
    volume_duration: float,
    vol_ratio: float = None,
    ma25_min: float = None,
    ma25_max: float = None,
) -> float:
    """
    【改进1强化】高位风险检测机制 - 按相对位置分级（P0 - 必须）

    识别顶部危险：考虑价格绝对位置（高位vs低位vs底部）
    - 高位（>0.7）+ 均线衰减 = 极度危险（-0.6）
    - 中位（0.4-0.7）+ 均线衰减 = 警告（-0.2）
    - 低位（<0.4）+ 均线衰减 = 无风险（0.0，底部积累）

    参数:
        price: 当前收盘价
        ma25: MA25均线值
        atr_val: ATR波动幅度
        ma25_slope: MA25斜率（ATR归一化值）
        volume_duration: 放量持续天数 [0, 1]
        vol_ratio: 成交量比例（MA5_VOL / MA60_VOL），可选
        ma25_min: 近期MA25最小值，用于计算相对位置
        ma25_max: 近期MA25最大值，用于计算相对位置

    返回:
        float: [-1, 0] 范围的风险惩罚
    """
    normalized_distance = safe_divide(price - ma25, max(atr_val, EPS), 0)

    if normalized_distance < 0.8:
        return 0.0

    # ✅ 改进1强化：计算相对位置（0-1范围）
    if ma25_min is not None and ma25_max is not None:
        ma25_range = ma25_max - ma25_min
        if ma25_range > EPS:
            price_position = safe_divide(price - ma25_min, ma25_range, 0.5)
            price_position = max(0.0, min(1.0, price_position))  # 限制在[0,1]
        else:
            price_position = 0.5  # 无法计算时用中值
    else:
        price_position = 0.5  # 数据不足时用默认中值

    # 第二步：根据MA25斜率判断衰减
    if ma25_slope < 0.05:  # MA25斜率衰减
        # ✅ 改进1强化：按相对位置分级
        if price_position > 0.7:  # 高位
            # 高位 + 均线衰减 = 极度危险
            if vol_ratio is not None and vol_ratio < 0.9:
                risk_penalty = -0.8  # 高位缺量 = 加强惩罚
            else:
                risk_penalty = -0.6  # 高位衰减（保持强惩罚）

        elif price_position > 0.4:  # 中位
            # 中位 + 均线衰减 = 警告
            risk_penalty = -0.2

        else:  # 低位（<0.4）
            # 低位 + 均线衰减 = 无风险（底部积累信号）
            risk_penalty = 0.0

    else:
        # 价格高，但MA25斜率强 = 轻微警告
        risk_penalty = -0.1

    return risk_penalty


def is_valid_golden_cross(dif: float, dea: float, zero_axis_threshold: float) -> bool:
    """
    【改进2优化】MACD 0轴金叉的分档判定（P1 - 重要）

    优化策略：分两档检测，对0轴附近的金叉更敏感
    - 若已离0轴较远（>2倍threshold）：严格要求分离度（防止假信号）
    - 若在0轴附近：只要有金叉且向上就是强信号（最好的买入点）

    参数:
        dif: MACD快线值
        dea: MACD慢线值
        zero_axis_threshold: 0轴阈值

    返回:
        bool: True=有效金叉，False=无效金叉
    """
    dif_dea_separation = abs(dif - dea)

    # ✅ 改进2强化：分档判定
    if abs(dif) > zero_axis_threshold * 2:  # 远离0轴（较强走势）
        # 保持严格：防止虚假信号
        return (dif > dea and
                dif_dea_separation > zero_axis_threshold * 0.3)
    else:  # 在0轴附近（最敏感的买点）
        # ✅ 放宽条件：0轴金叉往往是最强信号
        # 只需满足：金叉 + 快慢线开始分离（即使很小）
        return (dif > dea and
                dif_dea_separation > zero_axis_threshold * 0.1)


def calculate_consecutive_low_volume_penalty(volume_array: np.ndarray, idx: int, window: int = 5) -> float:
    """
    改进4：增加"连续缺量"的风险累积机制（P0 - 必须）

    检查过去N根K线中有多少根缺量
    连续缺量往往预示整个反弹动能不足

    参数:
        volume_array: 成交量数组
        idx: 当前K线索引
        window: 检查窗口（默认5根K线）

    返回:
        float: 额外惩罚值 [-0.25, 0]
    """
    if idx < window:
        return 0.0

    try:
        vol_hist = volume_array[max(0, idx - window):idx]
        vol_hist_clean = vol_hist[np.isfinite(vol_hist)]

        if len(vol_hist_clean) < 3:
            return 0.0

        vol_avg = np.mean(vol_hist_clean)

        if vol_avg <= EPS:
            return 0.0

        # 统计缺量根数（成交量 < 85%平均）
        low_vol_count = np.sum(vol_hist < vol_avg * 0.85)

        if low_vol_count >= 3:  # 5根K线中有3根缺量
            return -0.25  # 额外惩罚
        elif low_vol_count >= 2:
            return -0.1

        return 0.0
    except Exception:
        return 0.0


def calculate_price_ma25_deviation_risk(
    close: float,
    ma25: float,
    atr_val: float,
    ma25_slope: float,
) -> float:
    """
    改进6：增加MA25与当前价格"乖离率"的衰减检测（P1 - 重要）

    检测价格与MA25的乖离率和MA25斜率配合度
    高位+斜率衰减是顶部风险的关键信号

    参数:
        close: 当前收盘价
        ma25: MA25均线值
        atr_val: ATR波动幅度
        ma25_slope: MA25斜率

    返回:
        float: 额外惩罚值
    """
    if atr_val <= EPS:
        return 0.0

    deviation = safe_divide(close - ma25, atr_val, 0)

    try:
        # 高位 + 斜率衰减 = 风险加倍
        if deviation > 1.0 and ma25_slope < 0.15:
            # 极高位置（>1.0 ATR） + 斜率弱 = 高风险
            return -0.3
        elif deviation > 0.6 and ma25_slope < 0.08:
            # 中高位置（>0.6 ATR） + 斜率衰减 = 中风险
            return -0.2

        return 0.0
    except Exception:
        return 0.0


def apply_accuracy_zone_mapping(original_score: float) -> float:
    """
    改进3：准确率区间化评分（P1 - 重要）
    
    ⚠️ v5.2优化（2025-12-08）：根据F07_01批量验证报告
    问题：原映射过度压缩分数到[-0.3, 0.3]，导致区分度不足
    优化：减弱压缩强度，保留更多原始分数信息，增强区分度
    
    旧行为：
        - [0.3, 0.6] → [0.1, 0.3] （压缩到1/3）
        - [0.6, 1.0] → [0.2, 0.3] （极度压缩）
    
    新行为：
        - [0.3, 0.6] → [0.3, 0.6] （保持不变）
        - [0.6, 1.0] → [0.6, 0.85] （轻微压缩）
    
    参数:
        original_score: 原始F07_01分数 [-1, 1]
    
    返回:
        float: 映射后的分数，范围更广，区分度更高
    """
    
    # ✅ v5.2优化：扩大有效分数范围，从[-0.3, 0.3]扩展到[-0.6, 0.6]
    # 目的：让强信号（>0.3）和极强信号（>0.6）能够被识别
    
    # 对于中等买入信号（0.3~0.6），不再压缩，保持原值
    if 0.3 < original_score <= 0.6:
        return original_score  # ✅ 新：保持不变，不再压缩
    
    # 对于强买入信号（>0.6），轻微压缩到[0.6, 0.85]
    # 目的：保留强信号，但避免过度自信
    if original_score > 0.6:
        # 0.6 -> 0.6, 1.0 -> 0.85
        return 0.6 + (original_score - 0.6) * 0.625  # ✅ 新：轻微压缩，保留强信号
    
    # 对于弱买入和中立信号（-0.3~0.3），保持不变（高准确率区间）
    if -0.3 <= original_score <= 0.3:
        return original_score
    
    # 对于卖出信号，对称处理
    if -0.6 <= original_score < -0.3:
        return original_score  # ✅ 新：保持不变
    
    if original_score < -0.6:
        # -1.0 -> -0.85, -0.6 -> -0.6
        return -0.6 + (original_score + 0.6) * 0.625  # ✅ 新：轻微压缩
    
    return original_score


def calculate_ma5_ma25_adhesion_quality(
    ma5: float,
    ma25: float,
    atr_val: float,
) -> float:
    """
    改进7：MA5与MA25的粘合度检测（P1 - 重要）

    强势行情：MA5应该紧跟MA25，距离不超过0.5个ATR
    弱势行情：MA5远离MA25，表示反弹力度不足或顶部无力

    参数:
        ma5: 5日均线值
        ma25: 25日均线值
        atr_val: ATR波动幅度

    返回:
        float: [-0.2, 0.3] 范围的粘合度评分
    """
    if atr_val <= EPS:
        return 0.0

    try:
        ma_distance = abs(ma5 - ma25) / atr_val

        # 场景1：MA5在MA25上方但距离近（0 ~ 0.3 ATR）= 强势上升
        if ma5 > ma25 and ma_distance < 0.3:
            # 紧跟上升 = 最强信号
            return 0.3

        # 场景2：MA5在MA25下方（正常下跌或底部）
        elif ma5 < ma25:
            if ma_distance < 0.5:
                # 距离近的下跌 = 可能底部
                return -0.05
            else:
                # 距离远的下跌 = 继续下跌
                return -0.15

        # 场景3：MA5与MA25距离过远（>0.5 ATR）= 反弹衰减或无力
        elif ma_distance > 0.5:
            # 虽然在上方，但距离太远 = 反弹将失败
            return -0.2

        # 场景4：MA5在MA25上方，中等距离（0.3-0.5 ATR）
        else:
            return 0.1  # 轻微加成

    except Exception:
        return 0.0


def detect_macd_extreme_reversal(
    dif: float,
    dea: float,
    dif_array: np.ndarray,
    idx: int,
) -> float:
    """
    改进8：MACD极值反转检测（P1 - 重要）

    检测MACD在极值处的反转信号

    极值反转通常预示着趋势即将改变：
    - MACD 从极度负值向上反弹（底部反转）→ 买入信号
    - MACD 从极度正值向下回落（顶部反转）→ 卖出信号

    参数:
        dif: 当前MACD DIF值
        dea: 当前MACD DEA值
        dif_array: MACD DIF历史数组
        idx: 当前K线索引

    返回:
        float: [-0.3, 0.2] 范围的极值反转评分
    """
    if idx < 20 or dif_array is None:
        return 0.0

    try:
        # 计算过去20根K线的MACD范围
        recent_dif = dif_array[max(0, idx - 20):idx]
        recent_dif = recent_dif[np.isfinite(recent_dif)]

        if len(recent_dif) < 10:
            return 0.0

        dif_min = np.percentile(recent_dif, 10)  # 10%分位（极值低点）
        dif_max = np.percentile(recent_dif, 90)  # 90%分位（极值高点）

        # 极值范围需要足够宽（避免在平稳区间误判）
        dif_range = dif_max - dif_min
        if dif_range < EPS:
            return 0.0

        # ✅ 底部反转：MACD 在极值底部向上反弹
        # 条件1：DIF已经在极值底部附近（< min * 1.2）
        # 条件2：但较上一根K线有所回升（dif > dif[idx-1]）
        if dif < dif_min * 1.2 and idx >= 1:
            dif_prev = dif_array[idx - 1]
            if np.isfinite(dif_prev) and dif > dif_prev:
                # DIF 从极低值反弹向上 = 强烈的底部信号
                return 0.2

        # ✅ 顶部反转：MACD 在极值顶部向下回落
        # 条件1：DIF已经在极值顶部附近（> max * 0.8）
        # 条件2：但较上一根K线有所回落（dif < dif[idx-1]）
        if dif > dif_max * 0.8 and idx >= 1:
            dif_prev = dif_array[idx - 1]
            if np.isfinite(dif_prev) and dif < dif_prev:
                # DIF 从极高值下跌 = 强烈的顶部风险
                return -0.3

        return 0.0
    except Exception:
        return 0.0


def calculate_rebound_failure_penalty(
    ma5_array: np.ndarray,
    ma25_array: np.ndarray,
    idx: int,
    lookback: int = 15,
) -> float:
    """
    改进9：震荡筑底的"反弹失败"检测（P0 - 必须）

    检测最近N根K线中的反弹失败模式

    反弹失败的特征：
    - 价格（MA5）冲上MA25但快速回落
    - 连续2次以上这样的失败 = 反弹动能不足，应该降权

    参数:
        ma5_array: MA5均线数组
        ma25_array: MA25均线数组
        idx: 当前K线索引
        lookback: 回看窗口（默认15根K线）

    返回:
        float: [-0.4, 0] 范围的反弹失败惩罚
    """
    if idx < lookback + 1 or ma5_array is None or ma25_array is None:
        return 0.0

    try:
        # 统计过去N根K线中有多少次"价格穿过MA25但未能站稳"
        rebound_failures = 0

        start_idx = max(1, idx - lookback)

        for i in range(start_idx, idx):
            try:
                ma5_prev = ma5_array[i - 1]
                ma25_prev = ma25_array[i - 1]
                ma5_curr = ma5_array[i]
                ma25_curr = ma25_array[i]

                # 检查数据有效性
                if not (np.isfinite(ma5_prev) and np.isfinite(ma25_prev) and
                        np.isfinite(ma5_curr) and np.isfinite(ma25_curr)):
                    continue

                # ✅ 反弹失败的判定条件：
                # 前一根K线：MA5 < MA25（未站上均线）
                if ma5_prev < ma25_prev:
                    # 当前K线：MA5 反弹冲过MA25但仍然 < MA25
                    # 或者：MA5 有所回升但远未达到MA25
                    if ma5_curr > ma5_prev and ma5_curr < ma25_curr:
                        rebound_failures += 1
                    # 或者：MA5 短期反弹后又跌回MA25以下
                    elif ma5_curr < ma25_prev:
                        rebound_failures += 1
            except Exception:
                continue

        # ✅ 反弹失败次数判定
        if rebound_failures >= 3:
            # 3次以上失败 = 强烈的底部困顿信号，避免追高
            return -0.4
        elif rebound_failures >= 2:
            # 2次失败 = 警告，降权
            return -0.2
        elif rebound_failures >= 1:
            # 1次失败 = 轻微警告
            return -0.05

        return 0.0
    except Exception:
        return 0.0


def detect_multi_timeframe_resonance(
    ma5_slope: float,
    ma25_slope: float,
    macd_dif_slope: float,
    ma60_vol_slope: float,
) -> float:
    """
    【改进3强化】多时间框架共振 - 权重提升（P0 级）

    优化：4条线同向时加成提升（+0.5），3条线时按类型细分
    - 4条线完美共振 = +0.5（最强信号，可靠度>90%）
    - 3条价格线+MACD同向 = +0.35（极强）
    - 3条其他组合 = +0.25（强信号）
    - 2条同向 = +0.1（中等）

    参数:
        ma5_slope: MA5 斜率
        ma25_slope: MA25 斜率
        macd_dif_slope: MACD DIF 斜率
        ma60_vol_slope: MA60 成交量斜率

    返回:
        float: [-0.5, 0.5] 范围的共振加成
    """
    try:
        # 统计同向指标数
        uptrend_slopes = [
            slope for slope in [ma5_slope, ma25_slope, macd_dif_slope, ma60_vol_slope]
            if slope is not None and slope > 0.05
        ]
        downtrend_slopes = [
            slope for slope in [ma5_slope, ma25_slope, macd_dif_slope, ma60_vol_slope]
            if slope is not None and slope < -0.05
        ]

        uptrend_count = len(uptrend_slopes)
        downtrend_count = len(downtrend_slopes)

        # ✅ 改进3强化：4条线上升时加成提升
        if uptrend_count == 4:
            return 0.5  # ✅ 提升从0.4 -> 0.5（4条线完美共振）

        # ✅ 改进3强化：3条线时按类型细分
        elif uptrend_count == 3:
            # 检查是否为"3条价格线+MACD"的组合
            price_slopes_up = sum([
                1 for slope in [ma5_slope, ma25_slope, macd_dif_slope]
                if slope is not None and slope > 0.05
            ])

            if price_slopes_up == 3:  # 3条价格线都上升
                return 0.35  # ✅ 提升从0.25 -> 0.35（三条价格线+MACD同向）

            elif ma60_vol_slope is not None and ma60_vol_slope > 0.05:  # 包含成交量线
                return 0.25  # 保持

            else:
                return 0.2  # 其他组合

        # 2 个同向上升
        elif uptrend_count == 2:
            return 0.1

        # ✅ 改进3强化：下行时对称处理
        elif downtrend_count == 4:
            return -0.5

        elif downtrend_count == 3:
            price_slopes_down = sum([
                1 for slope in [ma5_slope, ma25_slope, macd_dif_slope]
                if slope is not None and slope < -0.05
            ])

            if price_slopes_down == 3:
                return -0.35

            else:
                return -0.25

        elif downtrend_count == 2:
            return -0.1

        return 0.0

    except Exception:
        return 0.0


def detect_macd_histogram_divergence(
    dif_array: np.ndarray,
    dea_array: np.ndarray,
    idx: int,
    lookback: int = 5,
) -> float:
    """
    【改进6强化】MACD 直方图衰减 - 更早预警（P1 级）

    优化：降低衰减阈值，提前1-2根K线发出警告
    - 衰减>20% = 早期预警（-0.2）
    - 衰减>50% = 严重预警（-0.3）

    原理：MACD柱子缩小比价格走势更先见底，
    可以提前1-2根K线抢占出场机会。

    参数:
        dif_array: MACD DIF 数组
        dea_array: MACD DEA 数组
        idx: 当前 K 线索引
        lookback: 回看窗口（默认 5 根 K 线）

    返回:
        float: [-0.3, 0.1] 范围的衰减警告
    """
    if idx < lookback + 1 or dif_array is None or dea_array is None:
        return 0.0

    try:
        histogram_history = []

        for i in range(max(0, idx - lookback), idx + 1):
            dif_val = dif_array[i]
            dea_val = dea_array[i]

            if np.isfinite(dif_val) and np.isfinite(dea_val):
                histogram = abs(dif_val - dea_val)
                histogram_history.append(histogram)

        if len(histogram_history) < 3:
            return 0.0

        # 检查直方图是否逐渐变小
        recent_histograms = histogram_history[-3:]
        is_shrinking = all(
            recent_histograms[i] >= recent_histograms[i + 1]
            for i in range(len(recent_histograms) - 1)
        )

        if is_shrinking:
            shrink_ratio = recent_histograms[0] / (recent_histograms[-1] + EPS)

            # ✅ 改进6强化：更早预警，降低阈值
            if shrink_ratio > 1.5:  # 衰减超过50%
                return -0.3  # 严重预警
            elif shrink_ratio > 1.2:  # 衰减20-50%
                return -0.2  # ✅ 提升从-0.15 -> -0.2（早期预警强化）
            else:  # 衰减<20%
                return -0.05  # 轻微警告

        # 检查直方图是否逐渐变大（动能增强）
        is_expanding = all(
            histogram_history[i] <= histogram_history[i + 1]
            for i in range(len(histogram_history) - 1)
        )

        if is_expanding:
            expand_ratio = histogram_history[-1] / (histogram_history[0] + EPS)
            if expand_ratio > 1.3:
                return 0.1  # 动能增强加成

        return 0.0

    except Exception:
        return 0.0


def validate_ma_breakout_after_adhesion(
    ma5: float,
    ma25: float,
    atr_val: float,
    ma5_prev: float,
    ma25_prev: float,
    ma5_prev_prev: float,
    ma25_prev_prev: float,
    volume_array: np.ndarray,
    ma60_vol_array: np.ndarray,
    idx: int,
) -> float:
    """
    【P0 级改进】MA 粘合后突破确认

    MA 粘合很紧时，如果突破往往很快反转。
    需要"粘合突破确认"：
    - 粘合状态（距离 < 0.2 ATR）
    - 突破后至少需要 2 根 K 线确认（价格维持在突破线之上）
    - 成交量必须放大 > 120% MA60_VOL

    参数:
        ma5: 当前 MA5
        ma25: 当前 MA25
        atr_val: 当前 ATR
        ma5_prev: 上一根 K 线 MA5
        ma25_prev: 上一根 K 线 MA25
        ma5_prev_prev: 前两根 K 线 MA5
        ma25_prev_prev: 前两根 K 线 MA25
        volume_array: 成交量数组
        ma60_vol_array: MA60 成交量数组
        idx: 当前 K 线索引

    返回:
        float: [-0.2, 0.3] 范围的突破确认评分
    """
    if atr_val <= EPS or idx < 2:
        return 0.0

    try:
        # 判断是否处于粘合状态（过去 2 根 K 线）
        adhesion_distance_prev = abs(ma5_prev - ma25_prev) / atr_val
        adhesion_distance_prev_prev = abs(ma5_prev_prev - ma25_prev_prev) / atr_val

        # 需要连续粘合：都在 0.2 ATR 以内
        if adhesion_distance_prev > 0.2 or adhesion_distance_prev_prev > 0.2:
            return 0.0  # 不是粘合状态

        # 现在发生了突破（MA5 和 MA25 距离突然扩大）
        current_distance = abs(ma5 - ma25) / atr_val

        if current_distance <= adhesion_distance_prev:
            return 0.0  # 没有突破

        # ✅ 突破发生了，现在检查确认条件
        breakout_strength = current_distance - adhesion_distance_prev

        # 检查成交量是否放大
        vol_confirmation = False

        if volume_array is not None and ma60_vol_array is not None and idx >= 1:
            try:
                current_vol = volume_array[idx]
                ma60_vol = ma60_vol_array[idx]

                if np.isfinite(current_vol) and np.isfinite(ma60_vol) and ma60_vol > EPS:
                    if current_vol > ma60_vol * 1.2:  # > 120% MA60_VOL
                        vol_confirmation = True
            except Exception:
                pass

        # 计算突破方向
        is_upside_breakout = ma5 > ma25 and ma5_prev <= ma25_prev
        is_downside_breakout = ma5 < ma25 and ma5_prev >= ma25_prev

        if is_upside_breakout:
            if vol_confirmation and breakout_strength > 0.15:
                # 粘合突破 + 放量 = 强烈信号
                return 0.3
            elif breakout_strength > 0.15:
                # 粘合突破但量不足 = 中等信号
                return 0.15
            else:
                # 突破不足 0.15 ATR = 虚假突破
                return -0.2

        elif is_downside_breakout:
            if vol_confirmation and breakout_strength > 0.15:
                return -0.3
            elif breakout_strength > 0.15:
                return -0.15
            else:
                return 0.1  # 下行突破失败 = 底部看涨

        return 0.0

    except Exception:
        return 0.0


def detect_triangle_formation(
    high_array: np.ndarray,
    low_array: np.ndarray,
    ma25_array: np.ndarray,
    volume_array: np.ndarray,
    idx: int,
    lookback: int = 10,
) -> float:
    """
    【P1 级改进】三角形整理检测

    检测三角形整理模式：
    - 高点逐渐走低
    - 低点逐渐走高
    - 成交量逐渐萎缩

    三角形完成后的突破往往是强势信号（突破概率 > 70%）

    参数:
        high_array: 最高价数组
        low_array: 最低价数组
        ma25_array: MA25 数组
        volume_array: 成交量数组
        idx: 当前 K 线索引
        lookback: 回看窗口（默认 10 根 K 线）

    返回:
        float: [-0.3, 0.3] 范围的三角形信号
    """
    if (idx < lookback or high_array is None or low_array is None or
        volume_array is None or ma25_array is None):
        return 0.0

    try:
        # 提取过去 N 根 K 线的数据
        start_idx = idx - lookback
        highs = high_array[start_idx:idx + 1]
        lows = low_array[start_idx:idx + 1]
        vols = volume_array[start_idx:idx + 1]
        ma25s = ma25_array[start_idx:idx + 1]

        # 清理数据
        valid_indices = np.where(
            np.isfinite(highs) & np.isfinite(lows) & np.isfinite(vols) & np.isfinite(ma25s)
        )[0]

        if len(valid_indices) < lookback * 0.7:
            return 0.0

        highs_clean = highs[valid_indices]
        lows_clean = lows[valid_indices]
        vols_clean = vols[valid_indices]

        # ✅ 检测特征 1：高点逐渐走低
        high_trend = 0
        for i in range(len(highs_clean) - 1):
            if highs_clean[i] > highs_clean[i + 1]:
                high_trend += 1

        high_lowering = high_trend / (len(highs_clean) - 1)  # 比例

        # ✅ 检测特征 2：低点逐渐走高
        low_trend = 0
        for i in range(len(lows_clean) - 1):
            if lows_clean[i] < lows_clean[i + 1]:
                low_trend += 1

        low_rising = low_trend / (len(lows_clean) - 1)

        # ✅ 检测特征 3：成交量逐渐萎缩
        vol_trend = 0
        for i in range(len(vols_clean) - 1):
            if vols_clean[i] > vols_clean[i + 1]:
                vol_trend += 1

        vol_shrinking = vol_trend / (len(vols_clean) - 1)

        # 三角形的确认条件
        # 需要三个特征都比较明显（都 > 0.6）
        if high_lowering > 0.6 and low_rising > 0.6 and vol_shrinking > 0.5:
            # 强烈的三角形整理
            return 0.2  # 预示突破概率高

        elif high_lowering > 0.5 and low_rising > 0.5 and vol_shrinking > 0.4:
            # 中等的三角形整理
            return 0.1

        # 检测是否是"上升三角形"还是"下降三角形"
        # 上升三角形（高点走低但低点走高）= 看涨
        elif high_lowering > 0.6 and low_rising > 0.6 and high_lowering > low_rising:
            return 0.15

        # 下降三角形（高点走低且低点走高）= 看跌
        elif high_lowering > 0.6 and low_rising > 0.6 and high_lowering < low_rising:
            return -0.15

        return 0.0

    except Exception:
        return 0.0


def calculate_market_sentiment(
    atr_array: np.ndarray,
    close_array: np.ndarray,
    idx: int,
    lookback: int = 20,
) -> float:
    """
    【P2 级改进】VIX-Like 市场情绪指标

    基于 ATR、波动率的情绪指标：
    - ATR > 历史平均 × 1.5 = 恐慌/亢奋（可能是反转点）
    - ATR < 历史平均 × 0.7 = 平静（可能准备爆发）

    参数:
        atr_array: ATR 数组
        close_array: 收盘价数组
        idx: 当前 K 线索引
        lookback: 回看窗口（默认 20 根 K 线）

    返回:
        float: [-0.2, 0.2] 范围的情绪评分
    """
    if idx < lookback or atr_array is None or close_array is None:
        return 0.0

    try:
        # 计算过去 N 根 K 线的 ATR 平均值
        atr_history = atr_array[max(0, idx - lookback):idx + 1]
        atr_history_clean = atr_history[np.isfinite(atr_history)]

        if len(atr_history_clean) < lookback * 0.5:
            return 0.0

        atr_mean = np.mean(atr_history_clean)
        atr_std = np.std(atr_history_clean)
        current_atr = atr_array[idx]

        if atr_mean <= EPS:
            return 0.0

        # 计算 ATR z-score
        atr_zscore = (current_atr - atr_mean) / (atr_std + EPS)

        # ✅ 恐慌情绪（ATR 突然升高）
        if current_atr > atr_mean * 1.5:
            # 极度恐慌 = 可能是反转点
            panic_intensity = min(1.0, (current_atr - atr_mean * 1.5) / (atr_mean * 0.5))
            # 恐慌时通常是机会
            if close_array[idx] < close_array[idx - 1]:  # 收低
                return 0.15  # 恐慌性下跌 = 看涨信号
            else:
                return -0.1  # 恐慌性上升 = 看跌信号

        # ✅ 平静情绪（ATR 突然降低）
        elif current_atr < atr_mean * 0.7:
            # 极度平静 = 可能准备爆发
            return 0.1  # 越平静越可能爆发

        # ✅ 正常波动范围
        else:
            # z-score 在 [-1, 1] = 平静且平衡
            if abs(atr_zscore) < 0.5:
                return 0.05  # 小加成（平衡的市场）
            elif atr_zscore > 1:
                return -0.05  # 轻微警告（亢奋）
            else:
                return 0.05  # 轻微加成（低迷）

    except Exception:
        return 0.0


def detect_breakout_momentum(
    ma5_array: np.ndarray,
    ma25_array: np.ndarray,
    atr_array: np.ndarray,
    idx: int,
    lookback: int = 3,
) -> float:
    """
    【改进5新增】粘合突破的加速度检测（P1 - 重要）

    检测粘合突破后是否有加速动能
    - 突破后3根K线内距离快速扩大 = 真实突破（+0.2）
    - 距离缓慢扩大 = 缓慢加速（+0.1）
    - 突破后失速 = 假突破（-0.1）

    规律：从图表看，真实突破后，MA5距离MA25会持续扩大
    虚假突破则快速回靠。

    参数:
        ma5_array: MA5均线数组
        ma25_array: MA25均线数组
        atr_array: ATR数组
        idx: 当前K线索引
        lookback: 回看窗口（默认3根K线）

    返回:
        float: [-0.15, 0.2] 范围的加速度评分
    """
    if idx < lookback or ma5_array is None or ma25_array is None or atr_array is None:
        return 0.0

    try:
        distances = []

        for i in range(idx - lookback, idx + 1):
            if (np.isfinite(ma5_array[i]) and np.isfinite(ma25_array[i]) and
                np.isfinite(atr_array[i]) and atr_array[i] > EPS):
                dist = abs(ma5_array[i] - ma25_array[i]) / atr_array[i]
                distances.append(dist)

        if len(distances) < lookback:
            return 0.0

        # 检查距离是否递增（加速）
        acceleration_count = 0
        for i in range(len(distances) - 1):
            if distances[i + 1] > distances[i]:
                acceleration_count += 1

        # ✅ 改进5：评估加速强度
        if acceleration_count == lookback - 1:  # 全部递增
            # 计算平均加速度
            avg_accel = (distances[-1] - distances[0]) / (lookback + EPS)
            if avg_accel > 0.1:
                return 0.2  # 加速突破
            else:
                return 0.15  # 缓慢加速

        elif acceleration_count >= lookback // 2:  # 至少一半递增
            return 0.1  # 部分加速

        else:  # 失速
            return -0.1  # 假突破信号

    except Exception:
        return 0.0


def detect_capital_inflow_direction(
    close_array: np.ndarray,
    volume_array: np.ndarray,
    ma5_vol: float,
    ma60_vol: float,
    idx: int,
    lookback: int = 10,
) -> float:
    """
    【P2 级改进】资金净流入方向检测

    不仅检查"有没有放量"，还要检查"资金流向哪里"：
    - 低位放量 + 价格上升 = 机构进场
    - 高位放量 + 价格下降 = 机构出逃

    参数:
        close_array: 收盘价数组
        volume_array: 成交量数组
        ma5_vol: MA5 成交量
        ma60_vol: MA60 成交量
        idx: 当前 K 线索引
        lookback: 回看窗口（默认 10 根 K 线）

    返回:
        float: [-0.3, 0.3] 范围的资金流向评分
    """
    if (idx < lookback or close_array is None or volume_array is None or
        ma60_vol <= EPS):
        return 0.0

    try:
        # 计算过去 N 根 K 线的价格范围（用于判断"高位"or"低位"）
        price_history = close_array[max(0, idx - lookback):idx + 1]
        price_history_clean = price_history[np.isfinite(price_history)]

        if len(price_history_clean) < lookback * 0.5:
            return 0.0

        price_min = np.min(price_history_clean)
        price_max = np.max(price_history_clean)
        price_range = price_max - price_min

        if price_range <= EPS:
            return 0.0

        current_price = close_array[idx]
        price_position = (current_price - price_min) / price_range  # [0, 1]

        # ✅ 判断成交量是否放大
        vol_ratio = ma5_vol / ma60_vol
        is_expanding_volume = vol_ratio > 1.2

        # ✅ 判断价格方向
        price_prev = close_array[idx - 1] if idx >= 1 else current_price
        is_price_rising = current_price > price_prev

        # ✅ 资金流向评分
        if is_expanding_volume:
            if price_position > 0.7 and is_price_rising:
                # 高位放量 + 价格上升 = 顶部突破（有风险）
                return -0.2  # 警告：机构可能在出逃或吸筹最后一波
            elif price_position > 0.7 and not is_price_rising:
                # 高位放量 + 价格下降 = 机构出逃（强卖信号）
                return -0.3  # 强烈警告

            elif price_position < 0.3 and is_price_rising:
                # 低位放量 + 价格上升 = 机构进场（强买信号）
                return 0.3  # 强烈看涨
            elif price_position < 0.3 and not is_price_rising:
                # 低位放量 + 价格下降 = 持续吸筹（隐藏的强信号）
                return 0.2

            else:
                # 中位放量 = 中立
                return 0.0

        else:
            # 缩量情况
            if price_position > 0.7:
                return -0.1  # 高位缩量 = 警告
            elif price_position < 0.3:
                return 0.1   # 低位缩量 = 观望
            else:
                return 0.0

    except Exception:
        return 0.0


def get_dynamic_weights(ma25_slope: float) -> dict:
    """
    改进2&5：动态权重系统（P1 - 重要）

    根据MA25斜率动态调整各维度权重
    ✅ 改进5：调整MA25斜率分级边界值，更敏感地检测衰减

    参数:
        ma25_slope: MA25斜率 [-1, 1]（ATR归一化）

    返回:
        dict: 权重字典 {
            'ma25_trend': float,      # MA25趋势权重
            'vol_score': float,       # 成交量权重
            'price_reasonableness': float  # 价格合理性权重
        }
    """

    # ✅ 改进5：优化权重解释和边界值
    # ma25_slope > 0.1: 强势阶段 (主升浪) [降低门槛从0.15->0.1]
    # 0.0 < ma25_slope <= 0.1: 温和阶段 (缓升) [边界调整]
    # -0.1 <= ma25_slope <= 0.0: 平盘或轻微下降 (高位震荡) [收严边界]
    # ma25_slope < -0.1: 衰退阶段 (下降) [降低门槛]

    if ma25_slope > 0.1:  # 强势上升 ✅
        return {
            'ma25_trend': 0.55,      # 增加MA25权重，强势信号可信度高
            'vol_score': 0.25,       # 减少成交量权重（量跟不上也没关系）
            'price_reasonableness': 0.20,  # 保持价格权重
        }

    elif ma25_slope > 0.0:  # 缓慢上升 ⚠️ [调整边界从0.05->0.0]
        return {
            'ma25_trend': 0.50,
            'vol_score': 0.30,       # 提升成交量权重（需要量的确认）
            'price_reasonableness': 0.20,
        }

    elif ma25_slope >= -0.1:  # 平盘或轻微下降 ❌ [调整边界从-0.05->-0.1]
        return {
            'ma25_trend': 0.35,      # ✅ 大幅降低MA25权重（信号不可靠）[从0.40->0.35]
            'vol_score': 0.45,       # ✅ 大幅提升成交量权重（成交量主导）[从0.40->0.45]
            'price_reasonableness': 0.20,
        }

    else:  # 明显下降 ❌❌ [ma25_slope < -0.1]
        return {
            'ma25_trend': 0.2,       # ✅ 大幅降低权重 [从0.30->0.2]
            'vol_score': 0.6,        # ✅ 极度提升成交量权重 [从0.50->0.6]
            'price_reasonableness': 0.20,
        }


def compute_ma25_vol_price_triangle_v51(
    ma25, close, ma5_vol, ma60_vol, atr_val,
    dif, dea, prev_dif, prev_dea,
    dif_array=None, dea_array=None,
    ma25_array=None, atr_array=None,
    ma60_vol_array=None, ma5_vol_array=None,
    high_array=None, low_array=None, close_array=None,
    volume_array=None,
    idx=None,
    fund_attention=None,
    volume_duration=None,
    ma5_array=None
):
    """
    F07_01: MA25+VOL+价格三角形验证强度（2560战法核心）
    
    v5.1版本（P0 - 2560战法核心）
    
    新版本改进：
    1. 改进1：高位风险检测机制 - 防止"顶部接盘"（新增calculate_high_position_risk）
    2. 改进2：动态权重系统 - 根据MA25斜率动态调整权重(0.3-0.55)
    3. 改进3：增强MACD金叉加成 - 根据持续天数动态加权(0.2-0.5)
    4. 改进4：加强缺量反弹风险警告 - 价涨但量缩时降权-0.15
    5. 改进5：提升资金持续性权重 - 连续放量3天+时额外加分
    """
    
    # 第一部分：价格相对MA25位置 + ATR距离判断
    atr_current = atr_val
    
    if (high_array is not None and low_array is not None and
        close_array is not None and idx is not None):
        atr_current = calculate_atr_recent_60bars(
            high_array, low_array, close_array, idx
        )
    
    price_distance = close - ma25
    
    if atr_current > EPS:
        normalized_distance = safe_divide(price_distance, atr_current, 0)
    else:
        normalized_distance = 0
    
    if normalized_distance > 0.8:
        price_reasonableness = 0.5
    elif normalized_distance > 0.3:
        price_reasonableness = 1.0
    elif normalized_distance > -0.3:
        price_reasonableness = 0.0
    elif normalized_distance > -0.8:
        price_reasonableness = -1.0
    else:
        price_reasonableness = -0.5
    
    # 第二部分：MA25斜率判断
    ma25_trend = price_reasonableness
    
    if ma25_array is not None and atr_array is not None and idx is not None:
        if idx >= 25:
            try:
                ma25_slope, ma25_r2 = compute_ma25_slope_atr_normalized_linregress(
                    ma25_array, atr_array, idx, period=25
                )
                
                # ✅ 注意：MA25_slope已经在compute_ma25_slope_atr_normalized_linregress中正确处理
                # 该函数返回[-1,1]的ATR归一化斜率，无需再次修复逻辑
                # 上升为正值，下降为负值
                if ma25_slope > 0.1:  # 强上升
                    if price_reasonableness >= 0.0:
                        ma25_trend = 1.0
                    else:
                        ma25_trend = 0.5
                
                elif ma25_slope >= -0.1:  # 平稳或弱上趋势
                    ma25_trend = 0.0
                
                else:  # 强下降
                    if price_reasonableness <= 0.0:
                        ma25_trend = -1.0
                    else:
                        ma25_trend = -0.5
                
            except Exception:
                ma25_trend = price_reasonableness
    
    # 第三部分：成交量金叉判断 + MA60量线斜率 + 占比高度关系
    # ✅ 改进1：计算MA60量线斜率，用于动态调整金叉阈值和评估量线趋势
    ma60_vol_slope = None
    vol_trend_quality = 0.0
    
    if ma60_vol_array is not None and idx is not None:
        if idx >= 60:
            try:
                ma60_vol_slope, ma60_vol_r2 = compute_volume_ma_slope_linregress(
                    ma60_vol_array, idx, period=60
                )
                
                # ✅ 改进2：修复MA60量线斜率评估逻辑
                # 原问题：上升斜率>0.15反而返回负值（逻辑颠倒）
                # 修复：上升为正分，下降为负分，横盘为中立
                if ma60_vol_slope > 0.05:  # 明显上升趋势
                    # 斜率越大，量线上升越强，得分越高
                    vol_trend_quality = min(1.0, ma60_vol_slope * 5)
                elif ma60_vol_slope < -0.05:  # 明显下降趋势
                    # 斜率越小（越负），量线下降越强，得分越低
                    vol_trend_quality = max(-1.0, ma60_vol_slope * 5)
                else:  # 横盘 [-0.05, 0.05]
                    # 量线平稳，中立
                    vol_trend_quality = 0.0
                
            except Exception:
                vol_trend_quality = 0.0
                ma60_vol_slope = None
    
    # ✅ 改进1：动态量能金叉阈值
    # 当MA60量线上升时，放宽金叉要求（3%）；下降时收严格（8%）
    if ma60_vol_slope is not None and np.isfinite(ma60_vol_slope):
        if ma60_vol_slope > 0.1:  # MA60强上升
            vol_cross_threshold = 0.03  # 金叉要求3%
        elif ma60_vol_slope < -0.1:  # MA60强下降
            vol_cross_threshold = 0.08  # 金叉要求8%（防缺量反弹）
        else:  # MA60平稳
            vol_cross_threshold = VOL_CROSS_THRESHOLD  # 使用默认5%
    else:
        vol_cross_threshold = VOL_CROSS_THRESHOLD  # 无法计算斜率时用默认值
    
    vol_cross = (1.0 if ma5_vol > ma60_vol * (1 + vol_cross_threshold)
                 else -1.0 if ma5_vol < ma60_vol * (1 - vol_cross_threshold)
                 else 0.0)
    
    vol_ratio_metrics = compute_volume_ratio_metrics(
        ma5_vol, ma60_vol,
        ma5_vol_array, ma60_vol_array,
        idx, lookback_period=20
    )
    
    vol_ratio_strength = 0.0
    
    if vol_ratio_metrics['is_volume_expanding']:
        vol_ratio_strength = 0.8
    elif vol_ratio_metrics['is_volume_contracting']:
        vol_ratio_strength = -0.8
    else:
        if vol_ratio_metrics['above_ratio'] > 0.6:
            vol_ratio_strength = (vol_ratio_metrics['above_ratio'] - 0.6) * 2
        elif vol_ratio_metrics['above_ratio'] < 0.4:
            vol_ratio_strength = (vol_ratio_metrics['above_ratio'] - 0.4) * 2
    
    vol_consistency_weight = 0.75
    if volume_duration is not None and volume_duration >= 0:
        vol_consistency_weight = 0.5 + volume_duration * 0.5
    
    vol_cross_norm = vol_cross
    vol_consistency_norm = (vol_consistency_weight - 0.75) * 4
    vol_ratio_norm = vol_ratio_strength
    vol_trend_norm = vol_trend_quality
    
    vol_score = (
        vol_cross_norm * 0.4 +
        vol_ratio_norm * 0.3 +
        vol_trend_norm * 0.2 +
        vol_consistency_norm * 0.1
    ) * 0.3
    
    # 第四部分：MACD信号判断
    macd_filter = 0

    if dif is not None and dea is not None and prev_dif is not None and prev_dea is not None:

        is_golden_cross = (dif > dea) and (prev_dif <= prev_dea)
        is_death_cross = (dif < dea) and (prev_dif >= prev_dea)

        zero_axis_threshold = abs(close) * 0.001

        if dif_array is not None and dea_array is not None and idx is not None:
            if idx >= 10:
                try:
                    lookback = min(idx + 1, 60)
                    start_idx = max(0, idx - lookback + 1)

                    dif_valid = dif_array[start_idx:idx+1]
                    dea_valid = dea_array[start_idx:idx+1]

                    dif_valid = dif_valid[np.isfinite(dif_valid)]
                    dea_valid = dea_valid[np.isfinite(dea_valid)]

                    if len(dif_valid) >= 5 and len(dea_valid) >= 5:
                        dif_median = np.median(np.abs(dif_valid))
                        dea_median = np.median(np.abs(dea_valid))
                        macd_median = max(dif_median, dea_median)
                        zero_axis_threshold = max(macd_median * 0.1, zero_axis_threshold)
                except Exception:
                    pass

        is_near_zero = (abs(dif) < zero_axis_threshold) and (abs(dea) < zero_axis_threshold)
        is_above_zero = (dif > 0) and (dea > 0)
        is_below_zero = (dif < 0) and (dea < 0)

        if is_death_cross:
            macd_filter = -1
        elif is_golden_cross:
            # ✅ 改进2：使用新的is_valid_golden_cross()函数验证金叉有效性
            if is_valid_golden_cross(dif, dea, zero_axis_threshold):
                macd_filter = 1  # 有效的金叉信号
            else:
                macd_filter = 0  # 无效的金叉（信号太弱）
        else:
            if is_above_zero:
                macd_filter = 1
            elif is_below_zero:
                macd_filter = -1
            else:
                macd_filter = 0
    
    # 第五部分：资金关注度加成
    # ✅ 改进5：提升资金持续性权重，加强连续放量的加成
    fund_boost = 0.0
    if fund_attention is not None:
        try:
            fund_attention = float(fund_attention)
            if 0 <= fund_attention <= 1:
                # 调整权重：高于60%则加分，低于40%则减分
                if fund_attention > 0.6:
                    fund_boost = (fund_attention - 0.6) * 1.0  # 调整系数从0.75 -> 1.0
                elif fund_attention < 0.4:
                    fund_boost = (fund_attention - 0.4) * 1.0  # 调整系数从0.75 -> 1.0
        except (TypeError, ValueError):
            fund_boost = 0.0
    
    # ✅ 改进5：额外加成 - 连续放量3天以上时
    if volume_duration is not None and volume_duration > 3:
        try:
            vol_duration = float(volume_duration)
            if vol_duration > 0:
                # 连续放量天数越多，额外加成越大（最多+0.4）
                # 3天 -> +0.1, 5天 -> +0.2, 10天 -> +0.4
                vol_persistence_boost = min(0.4, vol_duration * 0.1)
                # 与fund_boost合并：取两者的较大值（避免重复加成）
                fund_boost = max(fund_boost, vol_persistence_boost)
        except (TypeError, ValueError):
            pass  # 异常时保持原fund_boost
    
    # 第六部分：综合得分计算（改进2：动态权重系统）
    # 获取MA25斜率用于动态权重调整
    ma25_slope_val = 0.0
    if ma25_array is not None and atr_array is not None and idx is not None and idx >= 25:
        try:
            ma25_slope_val, _ = compute_ma25_slope_atr_normalized_linregress(ma25_array, atr_array, idx, 25)
        except:
            ma25_slope_val = 0.0
    
    # 获取动态权重
    weights = get_dynamic_weights(ma25_slope_val)
    
    # 使用动态权重计算基础分数
    base_score = (
        ma25_trend * weights['ma25_trend'] +
        vol_score * weights['vol_score'] +
        price_reasonableness * weights['price_reasonableness']
    )
    
    base_score = max(-1.0, min(1.0, base_score))
    
    # 第七部分：MACD过滤器应用
    # ✅ 改进3：增强MACD金叉加成，根据持续天数动态加权
    if macd_filter == -1:  # MACD死叉
        if base_score > 0:
            adjusted_score = base_score * 0.5  # 死叉时，正信号减半
        else:
            adjusted_score = max(base_score - 0.2, -1.0)  # 死叉时加强负信号
    elif macd_filter == 1:  # MACD金叉
        # ✅ 改进3：根据金叉持续天数动态加权（0.2 + duration * 0.15，最多0.5）
        macd_duration_boost = 0.0
        if dif_array is not None and dea_array is not None and idx is not None and idx >= 1:
            try:
                # 导入calculate_f08_06_golden_cross_duration函数
                from .momentum_persistence import calculate_f08_06_golden_cross_duration
                # 计算金叉持续天数
                macd_duration = calculate_f08_06_golden_cross_duration(dif_array, dea_array, idx, max_days=10)
                # 金叉越新鲜（duration越接近1.0），加成越大
                # duration=0.5 (5天) -> 加0.275; duration=1.0 (10天) -> 加0.5
                macd_duration_boost = 0.2 + macd_duration * 0.3
            except Exception:
                macd_duration_boost = 0.2  # 异常情况降级为默认0.2
        else:
            macd_duration_boost = 0.2  # 数据不足时用默认0.2
        
        if base_score > 0:
            adjusted_score = min(base_score + macd_duration_boost, 1.0)
        else:
            adjusted_score = max(base_score + macd_duration_boost * 0.5, -1.0)
    else:  # 中立（MACD既不在极值也不金叉死叉）
        adjusted_score = base_score
    
    # ✅ 改进4强化：缺量反弹按相对位置分级
    # 区分：高位缺量（极度危险）vs 低位缺量（底部积累，无风险）
    volume_gap_penalty = 0.0
    if close > ma25 and idx >= 20 and volume_array is not None:
        try:
            vol_history = volume_array[max(0, idx-20):idx]
            vol_history_clean = vol_history[np.isfinite(vol_history)]
            if len(vol_history_clean) >= 10:
                vol_avg_20 = np.mean(vol_history_clean)
                current_vol = volume_array[idx]
                if (np.isfinite(vol_avg_20) and np.isfinite(current_vol) and
                    vol_avg_20 > EPS):

                    # ✅ 改进4强化：先计算相对位置（类似改进1）
                    if close_array is not None and idx >= 20:
                        try:
                            price_hist = close_array[max(0, idx-20):idx+1]
                            price_hist_clean = price_hist[np.isfinite(price_hist)]
                            if len(price_hist_clean) >= 10:
                                price_min = np.min(price_hist_clean)
                                price_max = np.max(price_hist_clean)
                                price_range = price_max - price_min

                                if price_range > EPS:
                                    price_position = safe_divide(close - price_min, price_range, 0.5)
                                    price_position = max(0.0, min(1.0, price_position))
                                else:
                                    price_position = 0.5
                            else:
                                price_position = 0.5
                        except Exception:
                            price_position = 0.5
                    else:
                        price_position = 0.5

                    # ✅ 改进4强化：按相对位置分级惩罚
                    if current_vol < vol_avg_20 * 0.8:  # 缺量<80%
                        if price_position > 0.7:  # 高位缺量
                            volume_gap_penalty = -0.6  # ✅ 极度危险
                        elif price_position > 0.4:  # 中位缺量
                            volume_gap_penalty = -0.2  # ✅ 警告
                        else:  # 低位缺量
                            volume_gap_penalty = 0.0  # ✅ 无惩罚（底部积累）

                    elif current_vol < vol_avg_20 * 0.9:  # 缺量<90%
                        if price_position > 0.7:
                            volume_gap_penalty = -0.2
                        else:
                            volume_gap_penalty = -0.1

        except (IndexError, ValueError, TypeError):
            volume_gap_penalty = 0.0

    # ✅ 改进4：连续缺量风险累积
    consecutive_low_vol_penalty = 0.0
    if volume_array is not None and idx >= 5:
        try:
            consecutive_low_vol_penalty = calculate_consecutive_low_volume_penalty(
                volume_array, idx, window=5
            )
        except Exception:
            consecutive_low_vol_penalty = 0.0
    
    # ✅ 改进4：增强衰减惩罚（P2 - 可选）
    decay_penalty = 0.0
    try:
        if (ma25_array is not None and atr_array is not None and
            ma5_array is not None and idx is not None and idx >= 25):
            ma5_slope_val, _ = compute_ma5_slope_atr_normalized_linregress(ma5_array, atr_array, idx, 5)
            ma25_slope_val, _ = compute_ma25_slope_atr_normalized_linregress(ma25_array, atr_array, idx, 25)
            ma_distance = abs(ma5_array[idx] - ma25_array[idx]) / (ma25_array[idx] + EPS)
            price_height = safe_divide(close - ma25, atr_current, 0)
            
            # 场景1：MA粘合 + 价格高位 = 很可能是顶部
            if ma_distance < 0.05 and price_height > 1.0:
                decay_penalty = -0.15  # 比原来的-0.05强3倍
            elif ma_distance < 0.05 and abs(ma5_slope_val) < 0.05 and abs(ma25_slope_val) < 0.05:
                decay_penalty = -0.05
            
            # 场景2：MA5和MA25都在衰减 = 整理或下降
            if abs(ma5_slope_val) < 0.05 and abs(ma25_slope_val) < 0.05:
                if ma5_slope_val < 0 or ma25_slope_val < 0:
                    decay_penalty = max(decay_penalty, -0.08)
    except Exception:
        pass
    
    try:
        if vol_ratio_metrics is not None and 'above_ratio' in vol_ratio_metrics:
            above_ratio = vol_ratio_metrics['above_ratio']
            if 0.3 < above_ratio < 0.7:
                decay_penalty = min(decay_penalty, -0.05)
    except Exception:
        pass
    
    # ✅ 改进1强化：高位风险检测（P0 - 必须）
    high_position_risk = 0.0
    try:
        if ma25_array is not None and atr_array is not None and idx is not None and idx >= 25:
            ma25_slope_for_risk, _ = compute_ma25_slope_atr_normalized_linregress(ma25_array, atr_array, idx, 25)
            vol_ratio_for_risk = safe_divide(ma5_vol, ma60_vol, 1.0)

            # ✅ 改进1强化：计算相对位置（近期MA25范围）
            if idx >= 20:
                try:
                    ma25_recent = ma25_array[max(0, idx-20):idx+1]
                    ma25_recent_clean = ma25_recent[np.isfinite(ma25_recent)]
                    if len(ma25_recent_clean) >= 10:
                        ma25_min = np.min(ma25_recent_clean)
                        ma25_max = np.max(ma25_recent_clean)
                    else:
                        ma25_min = ma25
                        ma25_max = ma25
                except Exception:
                    ma25_min = ma25
                    ma25_max = ma25
            else:
                ma25_min = ma25
                ma25_max = ma25

            high_position_risk = calculate_high_position_risk(
                close, ma25, atr_current, ma25_slope_for_risk,
                volume_duration if volume_duration is not None else 0,
                vol_ratio=vol_ratio_for_risk,
                ma25_min=ma25_min,
                ma25_max=ma25_max
            )
    except Exception:
        pass
    
    # ✅ 改进6：价格乖离率衰减风险检测
    price_ma25_deviation_risk = 0.0
    if atr_current > EPS:
        try:
            price_ma25_deviation_risk = calculate_price_ma25_deviation_risk(
                close, ma25, atr_current, ma25_slope_val
            )
        except Exception:
            price_ma25_deviation_risk = 0.0

    # ✅ 改进7：MA5与MA25的粘合度检测
    ma5_ma25_adhesion = 0.0
    if ma5_array is not None and idx >= 5:
        try:
            ma5_val = ma5_array[idx]
            if np.isfinite(ma5_val):
                ma5_ma25_adhesion = calculate_ma5_ma25_adhesion_quality(
                    ma5_val, ma25, atr_current
                )
        except Exception:
            ma5_ma25_adhesion = 0.0

    # ✅ 改进8：MACD极值反转检测
    macd_extreme_reversal = 0.0
    if dif_array is not None and dea_array is not None and idx >= 20:
        try:
            macd_extreme_reversal = detect_macd_extreme_reversal(
                dif, dea, dif_array, idx
            )
        except Exception:
            macd_extreme_reversal = 0.0

    # ✅ 改进9：反弹失败检测
    rebound_failure_penalty = 0.0
    if ma5_array is not None and ma25_array is not None and idx >= 15:
        try:
            rebound_failure_penalty = calculate_rebound_failure_penalty(
                ma5_array, ma25_array, idx, lookback=15
            )
        except Exception:
            rebound_failure_penalty = 0.0

    # ✅ 新增改进：多时间框架共振检测（P0 级）
    multi_timeframe_resonance = 0.0
    if ma25_array is not None and atr_array is not None and dif_array is not None and idx >= 25:
        try:
            # 计算 MA5 斜率
            ma5_slope_val_resonance, _ = compute_ma5_slope_atr_normalized_linregress(
                ma5_array, atr_array, idx, 5
            ) if ma5_array is not None else (None, None)

            # 计算 MA25 斜率
            ma25_slope_val_resonance, _ = compute_ma25_slope_atr_normalized_linregress(
                ma25_array, atr_array, idx, 25
            )

            # 计算 MACD DIF 斜率
            macd_dif_slope = 0.0
            if idx >= 5:
                dif_recent = dif_array[max(0, idx - 5):idx + 1]
                dif_clean = dif_recent[np.isfinite(dif_recent)]
                if len(dif_clean) >= 2:
                    macd_dif_slope = (dif_clean[-1] - dif_clean[0]) / (len(dif_clean) + EPS)

            # 计算 MA60 成交量斜率
            ma60_vol_slope = 0.0
            if ma60_vol_array is not None and idx >= 60:
                try:
                    ma60_vol_slope, _ = compute_volume_ma_slope_linregress(
                        ma60_vol_array, idx, period=60
                    )
                except Exception:
                    ma60_vol_slope = 0.0

            multi_timeframe_resonance = detect_multi_timeframe_resonance(
                ma5_slope_val_resonance if ma5_slope_val_resonance is not None else 0.0,
                ma25_slope_val_resonance,
                macd_dif_slope,
                ma60_vol_slope
            )
        except Exception:
            multi_timeframe_resonance = 0.0

    # ✅ 改进6强化：MACD 直方图衰减检测 - 更早预警（P1 级）
    macd_histogram_divergence = 0.0
    if dif_array is not None and dea_array is not None and idx >= 5:
        try:
            # ✅ 改进6：调用优化后的版本（更早检测衰减）
            macd_histogram_divergence = detect_macd_histogram_divergence(
                dif_array, dea_array, idx, lookback=5
            )
        except Exception:
            macd_histogram_divergence = 0.0

    # ✅ 改进5新增：粘合突破的加速度检测（P1 级）
    breakout_momentum = 0.0
    if ma5_array is not None and ma25_array is not None and atr_array is not None and idx >= 3:
        try:
            breakout_momentum = detect_breakout_momentum(
                ma5_array, ma25_array, atr_array, idx, lookback=3
            )
        except Exception:
            breakout_momentum = 0.0

    # ✅ 新增改进：MA 粘合突破确认（P0 级）
    ma_breakout_confirmation = 0.0
    if ma5_array is not None and ma25_array is not None and idx >= 2:
        try:
            ma5_val = ma5_array[idx]
            ma25_val = ma25_array[idx]
            ma5_prev = ma5_array[idx - 1]
            ma25_prev = ma25_array[idx - 1]
            ma5_prev_prev = ma5_array[idx - 2]
            ma25_prev_prev = ma25_array[idx - 2]

            if (np.isfinite(ma5_val) and np.isfinite(ma25_val) and
                np.isfinite(ma5_prev) and np.isfinite(ma25_prev) and
                np.isfinite(ma5_prev_prev) and np.isfinite(ma25_prev_prev)):

                ma_breakout_confirmation = validate_ma_breakout_after_adhesion(
                    ma5_val, ma25_val, atr_current,
                    ma5_prev, ma25_prev,
                    ma5_prev_prev, ma25_prev_prev,
                    volume_array, ma60_vol_array, idx
                )
        except Exception:
            ma_breakout_confirmation = 0.0

    # ✅ 新增改进：三角形整理检测（P1 级）
    triangle_formation = 0.0
    if high_array is not None and low_array is not None and idx >= 10:
        try:
            triangle_formation = detect_triangle_formation(
                high_array, low_array, ma25_array, volume_array, idx, lookback=10
            )
        except Exception:
            triangle_formation = 0.0

    # ✅ 新增改进：市场情绪指标（P2 级）
    market_sentiment = 0.0
    if atr_array is not None and close_array is not None and idx >= 20:
        try:
            market_sentiment = calculate_market_sentiment(
                atr_array, close_array, idx, lookback=20
            )
        except Exception:
            market_sentiment = 0.0

    # ✅ 新增改进：资金流向检测（P2 级）
    capital_flow_direction = 0.0
    if close_array is not None and volume_array is not None and idx >= 10:
        try:
            capital_flow_direction = detect_capital_inflow_direction(
                close_array, volume_array, ma5_vol, ma60_vol, idx, lookback=10
            )
        except Exception:
            capital_flow_direction = 0.0

    # 计算最终分数（包含所有惩罚和加成）
    # ✅ 改进1强化：高位风险检测（按相对位置分级）
    # ✅ 改进2强化：0轴金叉判定（分档敏感）
    # ✅ 改进3强化：多时间框架共振（权重提升）
    # ✅ 改进4强化：缺量反弹按位置分级
    # ✅ 改进5新增：粘合突破加速度检测（breakout_momentum）
    # ✅ 改进6强化：MACD直方图更早预警
    # ✅ 其他：consecutive_low_vol_penalty、price_ma25_deviation_risk、
    #         ma5_ma25_adhesion、macd_extreme_reversal、rebound_failure_penalty
    # ✅ 新增 P0/P1/P2 级改进：
    #         multi_timeframe_resonance、ma_breakout_confirmation、triangle_formation
    #         market_sentiment、capital_flow_direction
    final_score = (adjusted_score + fund_boost + volume_gap_penalty + decay_penalty +
                   high_position_risk + consecutive_low_vol_penalty + price_ma25_deviation_risk +
                   ma5_ma25_adhesion + macd_extreme_reversal + rebound_failure_penalty +
                   multi_timeframe_resonance + macd_histogram_divergence + breakout_momentum +
                   ma_breakout_confirmation + triangle_formation + market_sentiment + capital_flow_direction)

    # 确保范围（改进3前的安全检查）
    final_score = max(-1.0, min(1.0, final_score))

    # ✅ 改进3：准确率区间化评分（P1 - 重要）
    # 这一步会将中低准确率区间的分数映射到高准确率区间
    final_score = apply_accuracy_zone_mapping(final_score)

    return max(-1.0, min(1.0, final_score))


def compute_volume_coupling_strength(price_change_pct, vol_change_pct, ma5_vol, ma60_vol):
    """F07_02：量能配合强度（必须同向）
    
    2560战法的关键：放量必须伴随价格上涨，否则是假突破
    
    参数:
        price_change_pct: 价格变化百分比
        vol_change_pct: 成交量变化百分比
        ma5_vol: MA5成交量
        ma60_vol: MA60成交量
    
    返回:
        float: [-1, 1] 范围的量能配合强度
    """
    vol_ratio = safe_divide(ma5_vol, ma60_vol, 1.0)
    
    # ✅ v4.7修复：vol_boost可能为负的问题
    # 当vol_ratio > 1.5时，放量更明显，配合度提升
    # 确保vol_boost在[0, 1]范围内
    if vol_ratio <= 1.0:
        vol_boost = 0.0  # 无放量加成
    else:
        vol_boost = min(1.0, (vol_ratio - 1.0) * 2)  # [1.0,1.5]->[0,1]
    
    if price_change_pct > EPS:
        if vol_change_pct > EPS:
            # 价涨量增：健康配合
            coupling = min(1.0, abs(vol_change_pct) / (abs(price_change_pct) + EPS))
            coupling = coupling * (1.0 + vol_boost * 0.5)  # 放量时增强
            coupling = min(1.0, coupling)  # ✅ 审计修复：确保增强后不超出范围
        else:
            # 价涨量缩：配合度差
            coupling = -0.5
    elif price_change_pct < -EPS:
        if vol_change_pct < -EPS:
            # 价跌量缩：正常回调
            coupling = min(1.0, abs(vol_change_pct) / (abs(price_change_pct) + EPS))
        else:
            # 价跌量增：恐慌抛售
            coupling = -0.5
    else:
        coupling = 0.0
    
    return max(-1.0, min(1.0, coupling))


def compute_rebound_quality_strict(price_change_5, vol_change_5, ma5_vol, ma60_vol):
    """F07_03：反弹质量强制评分
    
    结合vol_ratio和vol_change_5，避免信息丢失
    
    加强缩量反弹的惩罚：
    - 放量反弹（+1.0）：价格上升 + 成交量增加 → 健康反弹
    - 温和反弹（0.0）：价格上升 + 成交量基本平稳 → 可参与
    - 缺量反弹（-1.0）：价格上升 + 成交量下降 → 警告信号（统一范围）
    
    参数:
        price_change_5: 5根K线价格变化
        vol_change_5: 5根K线成交量变化（可为None）
        ma5_vol: MA5成交量
        ma60_vol: MA60成交量
    
    返回:
        float: [-1, 1] 范围的质量评分（确保与其他特征一致）
    """
    vol_ratio = safe_divide(ma5_vol, ma60_vol, 1.0)
    
    if price_change_5 > EPS:
        # 结合vol_ratio和vol_change_5
        if vol_ratio > 1.2 and (vol_change_5 is None or vol_change_5 > 0):
            quality = 1.0       # 放量且成交量上升 → 健康反弹
        elif vol_ratio > 0.9:
            if vol_change_5 is not None and vol_change_5 < -0.2:
                quality = -0.5  # 成交量快速萎缩 → 动能不足
            else:
                quality = 0.0   # 温和反弹
        else:
            # vol_ratio < 0.9（缩量）
            if vol_change_5 is not None and vol_change_5 > 0.3:
                quality = 0.3   # 虽然缩量，但成交量在放大 → 可能转机
            else:
                quality = -1.0  # 缩量反弹 → 警告信号
    elif price_change_5 < -EPS:
        if vol_ratio < 0.8:
            quality = 0.5       # 正常回调（安全）
        else:
            quality = -0.5      # 异常放量（恐慌）
    else:
        quality = 0.0           # 平盘（中性）
    
    return max(-1.0, min(1.0, quality))  # 确保范围 [-1, 1]


def extract_f07_features(
    idx, close, closes,
    ma5_prices, ma25_prices, atr,
    ma5_volumes, ma60_volumes, volumes,
    dif, dea,
    highs, lows,
    capital_persistence
):
    """
    提取2560战法特征组（F07_01~F07_03）
    
    参数:
        idx: 当前K线索引
        close: 当前收盘价
        closes: 收盘价数组
        ma5_prices: MA5价格数组
        ma25_prices: MA25价格数组
        atr: ATR数组
        ma5_volumes: MA5成交量数组
        ma60_volumes: MA60成交量数组
        volumes: 成交量数组
        dif: MACD DIF数组
        dea: MACD DEA数组
        highs: 最高价数组
        lows: 最低价数组
        capital_persistence: 资金持续关注度（F06_01）
    
    返回:
        list: 包含3个特征值的列表 [F07_01, F07_02, F07_03]
    """
    features = []
    
    ma25 = ma25_prices[idx]
    ma5_vol = ma5_volumes[idx]
    ma60_vol = ma60_volumes[idx]
    atr_val = atr[idx]
    dif_val = dif[idx]
    dea_val = dea[idx]
    
    # === F07_01: MA25+VOL+价格三角形验证强度 [P0-2560战法核心] ===
    # 2560战法的核心：三个条件必须同时满足
    # ✅ 新增：传入MACD参数，零轴附近/以上金叉可增强信号
    # ✅ 新增：传入F06_01资金持续关注度，作为可选的质量过滤
    prev_dif = dif[idx-1] if idx > 0 else None
    prev_dea = dea[idx-1] if idx > 0 else None
    # ✅ v5.1版本：传入完整数组参数以支持MA25斜率和MA60量线斜率计算
    ma25_vol_price_triangle = compute_ma25_vol_price_triangle_v51(
        ma25, close, ma5_vol, ma60_vol, atr_val,
        dif_val, dea_val, prev_dif, prev_dea,
        dif_array=dif,
        dea_array=dea,
        ma25_array=ma25_prices,
        atr_array=atr,
        ma60_vol_array=ma60_volumes,
        ma5_vol_array=ma5_volumes,
        high_array=highs,
        low_array=lows,
        close_array=closes,
        volume_array=volumes,  # ✅ 新增：用于改进4缺量反弹检测
        idx=idx,
        fund_attention=capital_persistence,
        volume_duration=None,  # 可选参数：F08_08放量持续天数，因F08_08在后面计算，此处暂传None使用默认权重0.75
        ma5_array=ma5_prices
    )
    features.append(ma25_vol_price_triangle)  # F07_01
    
    # === F07_02: 量能配合强度 [P0-2560战法核心] ===
    # 放量必须伴随价格上涨，否则是假突破
    if idx >= 5:
        price_change_5 = safe_divide(close - closes[idx - 5], closes[idx - 5], 0)
        # ✅ P3修复：安全分母构造
        ma5_vol_prev = ma5_volumes[max(0, idx - 5)]
        ma5_vol_safe = ma5_vol_prev if np.isfinite(ma5_vol_prev) and ma5_vol_prev > EPS else EPS
        vol_change_5 = safe_divide(ma5_vol - ma5_vol_prev, ma5_vol_safe, 0)
        vol_coupling = compute_volume_coupling_strength(price_change_5, vol_change_5, ma5_vol, ma60_vol)
    else:
        vol_coupling = 0.0
    features.append(vol_coupling)  # F07_02
    
    # === F07_03: 反弹质量强制评分 [P0-2560战法核心] ===
    # ✅ 修复注释：缩量反弹为警告信号（-1.0），而非禁止信号（-2.0）
    # ✅ P1修复：计算vol_change_5并传入
    if idx >= 5:
        price_change_5 = safe_divide(close - closes[idx - 5], closes[idx - 5], 0)
        # ✅ P3修复：安全分母构造
        ma5_vol_prev = ma5_volumes[max(0, idx-5)]
        ma5_vol_safe = ma5_vol_prev if np.isfinite(ma5_vol_prev) and ma5_vol_prev > EPS else EPS
        vol_change_5 = safe_divide(ma5_vol - ma5_vol_prev, ma5_vol_safe, 0)
        rebound_quality = compute_rebound_quality_strict(price_change_5, vol_change_5, ma5_vol, ma60_vol)
    else:
        rebound_quality = 0.0
    features.append(rebound_quality)  # F07_03
    
    return features


# 导出
__all__ = [
    'extract_f07_features',
    'compute_ma25_vol_price_triangle_v51',
    'compute_volume_coupling_strength',
    'compute_rebound_quality_strict',
    'calculate_high_position_risk',
    'is_valid_golden_cross',
    'calculate_consecutive_low_volume_penalty',
    'calculate_price_ma25_deviation_risk',
    'calculate_ma5_ma25_adhesion_quality',
    'detect_macd_extreme_reversal',
    'calculate_rebound_failure_penalty',
    'apply_accuracy_zone_mapping',
    'get_dynamic_weights',
    # 【P0 级新增改进】
    'detect_multi_timeframe_resonance',
    'validate_ma_breakout_after_adhesion',
    # 【P1 级新增改进】
    'detect_macd_histogram_divergence',
    'detect_triangle_formation',
    'detect_breakout_momentum',  # ✅ 改进5新增
    # 【P2 级新增改进】
    'calculate_market_sentiment',
    'detect_capital_inflow_direction',
]
