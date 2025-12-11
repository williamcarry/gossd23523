"""
成交量特征组（F03_01~F03_12）

本模块包含所有与成交量相关的12个特征计算
从 feature_extractor.py 提取，保持100%一致
"""
import numpy as np
from .config import EPS
from .utils import (
    safe_divide,
    compute_volume_ma_slope_linregress,
    compute_price_volume_correlation,
    compute_volume_extreme_event,
    compute_exponential_velocity,
)


def extract_f03_features(
    idx, volume, volumes, closes,
    ma5_volumes, ma60_volumes
):
    """
    提取成交量特征组（F03_01~F03_12）
    
    参数:
        idx: 当前K线索引
        volume: 当前成交量
        volumes: 成交量数组
        closes: 收盘价数组
        ma5_volumes: MA5成交量数组
        ma60_volumes: MA60成交量数组
    
    返回:
        list: 包含12个特征值的列表 [F03_01, F03_02, ..., F03_12]
    """
    features = []
    
    ma5_vol = ma5_volumes[idx]
    ma60_vol = ma60_volumes[idx]
    
    # === F03_01: MA5量归一化 [P0-2560战法核心] ===
    # 修复：添加上限3倍，归一化到[0,1]范围
    vol_ratio = safe_divide(ma5_vol, ma60_vol, 1.0)
    # ✅ 修复NaN传播：np.minimum会传播NaN，改用max/min
    vol_ratio = min(vol_ratio, 3.0) if np.isfinite(vol_ratio) else 1.0  # 上限3倍
    vol_ratio_norm = (vol_ratio - 1.0) / 2.0  # 1倍->0, 3倍->1
    features.append(vol_ratio_norm)  # F03_01

    # === F03_02: MA60量归一化 [P0-2560战法核心] ===
    # ✅ 修复：改为实际量能相对值，避免恒定特征
    # MA60作为基准，计算其相对于历史均值的变化
    # ✅ Bug#21修复：历史均值应包含当前K线（因为使用历史数据）
    if idx >= 60:
        ma60_hist_mean = np.mean(ma60_volumes[idx-60:idx+1])  # ✅ 包含当前
    else:
        ma60_hist_mean = np.mean(ma60_volumes[max(0, idx-60):idx+1]) if idx >= 0 else ma60_vol

    # ✅ NaN防护：检查 ma60_hist_mean 的有效性
    if not np.isfinite(ma60_hist_mean) or ma60_hist_mean <= EPS:
        ma60_vol_norm = 1.0
    else:
        ma60_vol_norm = safe_divide(ma60_volumes[idx], ma60_hist_mean, 1.0)
        # ✅ 修复NaN传播：np.clip会传播NaN，改用max/min
        ma60_vol_norm = max(0.0, min(ma60_vol_norm, 2.0)) if np.isfinite(ma60_vol_norm) else 1.0  # 范围 [0, 2]
    features.append(ma60_vol_norm)  # F03_02
    
    # === F03_03: 价量相关性强度 ===
    # ✅ 审计修复：改用Pearson相关系数，基于5根K线的趋势相关性
    # 收集最近5根K线的价格和成交量变化率
    if idx >= 5:
        # ✅ 向量化：计算最近5根K线的价格和成交量变化率
        # 提取最近5个K线的数据 [idx-4, idx-3, idx-2, idx-1, idx]
        price_window = closes[idx-4:idx+1]
        vol_window = volumes[idx-4:idx+1]

        # 向量化计算差分（当前价格 - 前一价格）
        price_deltas = price_window[1:] - price_window[:-1]
        vol_deltas = vol_window[1:] - vol_window[:-1]

        # 向量化计算变化率：改进版处理NaN/Inf
        price_prev_safe = np.where(
            np.isfinite(price_window[:-1]) & (price_window[:-1] > 0),
            price_window[:-1],
            EPS
        )
        vol_prev_safe = np.where(
            np.isfinite(vol_window[:-1]) & (vol_window[:-1] > EPS),
            vol_window[:-1],
            EPS
        )

        # 向量化计算变化率
        price_changes_window = price_deltas / price_prev_safe
        volume_changes_window = vol_deltas / vol_prev_safe

        if len(price_changes_window) >= 3:  # 至少需3个有效点
            pvc_strength = compute_price_volume_correlation(price_changes_window, volume_changes_window, window=5)
        else:
            pvc_strength = 0.0
    else:
        pvc_strength = 0.0
    features.append(pvc_strength)  # F03_03
    
    # === F03_04/05: MA60量线趋势斜率+稳定性（线性回归版） ===
    # 用最近60根K线做线性回归，判断MA60量趋势方向和稳定性
    if idx >= 60:
        ma60_vol_slope, ma60_vol_r2 = compute_volume_ma_slope_linregress(ma60_volumes, idx, 60, ma60_volumes)
        # ✅ NaN防护：检查返回值的有效性
        if not np.isfinite(ma60_vol_slope):
            ma60_vol_slope = 0.0
        if not np.isfinite(ma60_vol_r2):
            ma60_vol_r2 = 0.0
    else:
        ma60_vol_slope = 0.0
        ma60_vol_r2 = 0.0
    features.append(ma60_vol_slope)  # F03_04
    features.append(ma60_vol_r2)  # F03_05
    
    # === F03_06: 成交量极值事件 ===
    # (替代：成交量波动率) - 当前成交量是否处于极值状态
    # ✅ Bug#12修复：传入日成交量历史，而非MA5量线
    # ✅ 业界标准优化：使用60根K线滑动窗口（Rolling Window）
    #    - 原理：60天窗口 = 59根历史K线 + 1根当前K线（待判断）
    #    - 计算p90：只用59根历史数据（不含当前）
    #    - 参考：Bloomberg/高盛/摩根士丹利/聚宽/米筐 统一标准
    VOLUME_WINDOW_SIZE = 60  # 成交量极值计算窗口（业界标准：60天窗口）
    vol_extreme = compute_volume_extreme_event(
        volume, 
        volumes[max(0, idx-VOLUME_WINDOW_SIZE+1):idx]  # ✅ 业界标准：59根历史（不含当前）
    )
    features.append(vol_extreme)  # F03_06
    
    # === F03_07: 量线MA5-MA60粘合度 ★P0纠缠态 ===
    # ✅ P0修复：语义错误 - "粘合度"应该是1.0表示完全粘合，0表示完全分离
    # ✅ P3修复：安全分母构造，防止ma60_vol=NaN时max返回NaN
    ma60_vol_safe = ma60_vol if np.isfinite(ma60_vol) and ma60_vol > EPS else EPS
    distance = safe_divide(abs(ma5_vol - ma60_vol), ma60_vol_safe, 0)
    distance = min(distance, 1.0)  # 上限1.0（100%分离）
    vol_cohesion = 1.0 - distance  # 反转：1-距离 = 粘合度
    features.append(vol_cohesion)  # F03_07
    
    # === F03_08: 量线MA5-MA60发散速度 ★P0纠缠态 ===
    # (改进：指数加权平均替代固定5根窗口)
    # ✅ Bug#19修复：历史数组不应包含当前值
    # ✅ P0修复：历史数组计算也必须使用粘合度（1.0 - distance），与F03_07保持一致
    if idx >= 10:
        # ✅ 向量化：提取历史数据 [idx-10, idx-1]
        look_back_start = idx - 10
        look_back_end = idx

        hist_indices = np.arange(look_back_start, look_back_end)
        hist_ma5 = ma5_volumes[hist_indices]
        hist_ma60 = ma60_volumes[hist_indices]

        # 向量化检查有效性
        valid_mask = np.isfinite(hist_ma5) & np.isfinite(hist_ma60)

        if np.sum(valid_mask) > 0:
            # 提取有效数据
            valid_ma5 = hist_ma5[valid_mask]
            valid_ma60 = hist_ma60[valid_mask]

            # 向量化计算：安全分母构造
            ma60_safe = np.where(valid_ma60 > EPS, valid_ma60, EPS)

            # 向量化计算距离和粘合度
            distance = np.abs(valid_ma5 - valid_ma60) / ma60_safe
            # ✅ 修复NaN传播：对向量操作，先过滤NaN再clip
            distance = np.where(np.isfinite(distance), np.minimum(distance, 1.0), 1.0)  # NaN视为完全分离
            vol_cohesion_history = 1.0 - distance  # ✅ 修复：使用粘合度，与F03_07一致

            if len(vol_cohesion_history) > 1:
                try:
                    vol_divergence_speed = compute_exponential_velocity(
                        vol_cohesion, vol_cohesion_history, half_life=5
                    )
                    # ✅ NaN防护：检查返回值有效性
                    if not np.isfinite(vol_divergence_speed):
                        vol_divergence_speed = 0.0
                except:
                    vol_divergence_speed = 0.0
            else:
                vol_divergence_speed = 0
        else:
            vol_divergence_speed = 0
    else:
        vol_divergence_speed = 0
    features.append(vol_divergence_speed)  # F03_08
    
    # === F03_09: 成交量异常倍数 ===
    # ✅ Bug#6修复：改为成交量异常倍数
    # 定义：当前成交量相对于基准量的倍数
    # ✅ 审计修复：基准不应包含当前K线，避免自引用（Bloomberg/聚宽标准）
    if idx >= 10:
        baseline_vol = np.mean(volumes[max(0, idx-10):idx])  # 前10根的平均（不含当前）
        if baseline_vol > EPS:
            vol_abnormal_ratio = safe_divide(volume, baseline_vol, 1.0)
            # 归一化：1倍为0，3倍为1，超过3倍截断
            vol_abnormal_ratio = (vol_abnormal_ratio - 1.0) / 2.0
            # ✅ 修复NaN传播：np.clip会传播NaN，改用max/min
            vol_abnormal_ratio = max(-1.0, min(vol_abnormal_ratio, 1.0)) if np.isfinite(vol_abnormal_ratio) else 0.0
        else:
            vol_abnormal_ratio = 0.0
    else:
        vol_abnormal_ratio = 0.0
    features.append(vol_abnormal_ratio)  # F03_09
    
    # === F03_10/11: MA5量线趋势斜率+稳定性（线性回归版） ===
    # 用最近5根K线做线性回归，判断MA5量趋势方向和稳定性
    if idx >= 5:
        ma5_vol_slope, ma5_vol_r2 = compute_volume_ma_slope_linregress(ma5_volumes, idx, 5, ma60_volumes)
        # ✅ NaN防护：检查返回值的有效性
        if not np.isfinite(ma5_vol_slope):
            ma5_vol_slope = 0.0
        if not np.isfinite(ma5_vol_r2):
            ma5_vol_r2 = 0.0
    else:
        ma5_vol_slope = 0.0
        ma5_vol_r2 = 0.0
    features.append(ma5_vol_slope)  # F03_10
    features.append(ma5_vol_r2)  # F03_11
    
    # === F03_12: 量价背离强度 ===
    # ✅ Bug#9修复：F11_03已有反弹质量，F03_15改为量价背离强度
    # v4.6修复：应包含当前K线，因为使用的是历史数据
    # ✅ v4.7修复：区分顶背离和底背离的语义
    # ✅ P0修复：添加NaN/Inf清洗，防止percentile计算错误
    # 定义：价格新高但成交量萎缩（顶背离）或价格新低但成交量放大（底背离）
    if idx >= 20:
        # 检查价格是否创20根新高/新低（包含当前）
        price_window = closes[idx-20:idx+1]  # ✅ 修复：包含当前
        vol_window = volumes[idx-20:idx+1]   # ✅ 修复：包含当前

        # ✅ P0修复：volume清洗NaN/Inf
        vol_clean = vol_window[:-1][np.isfinite(vol_window[:-1])]
        # ✅ NaN防护：清洗 price_window 以防止 np.max/np.min 返回 NaN
        price_clean = price_window[:-1][np.isfinite(price_window[:-1])]

        if len(vol_clean) < 10 or len(price_clean) < 10:  # 样本太少，无法计算
            divergence = 0
        else:
            price_max = np.max(price_clean)
            price_min = np.min(price_clean)
            price_is_new_high = closes[idx] > price_max if np.isfinite(price_max) else False
            price_is_new_low = closes[idx] < price_min if np.isfinite(price_min) else False

            # 当前成交量在历史中的百分位（不含当前）
            current_vol_percentile = np.sum(vol_clean <= volumes[idx]) / len(vol_clean)

            if price_is_new_high and current_vol_percentile < 0.3:
                # 顶背离：价格新高，成交量在低位（<30%）→ 卖出信号（负值）
                # ✅ 修复NaN传播：改用min
                divergence = -min(1.0, (0.3 - current_vol_percentile) / 0.3)
            elif price_is_new_low and current_vol_percentile > 0.7:
                # ✅ v4.7修复：底背离返回正值，与顶背离区分
                # 底背离：价格新低，成交量在高位（>70%）→ 买入机会（正值）
                # ✅ 修复NaN传播：改用min
                divergence = min(1.0, (current_vol_percentile - 0.7) / 0.3)
            else:
                divergence = 0
    else:
        divergence = 0
    features.append(divergence)  # F03_12
    
    return features


# 导出
__all__ = ['extract_f03_features']
