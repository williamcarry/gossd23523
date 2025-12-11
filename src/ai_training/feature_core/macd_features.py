"""
MACD特征组（F02_01~F02_09）

本模块包含所有与MACD指标相关的9个特征计算
从 feature_extractor.py 提取，保持100%一致
"""
import numpy as np
from .config import EPS
from .utils import safe_divide, compute_exponential_velocity


def extract_f02_features(
    idx, close, closes,
    dif, dea, macd_histogram
):
    """
    提取MACD特征组（F02_01~F02_09）

    参数:
        idx: 当前K线索引
        close: 当前收盘价
        closes: 收盘价数组
        dif: DIF数组
        dea: DEA数组
        macd_histogram: MACD柱状图数组

    返回:
        list: 包含9个特征值的列表 [F02_01, F02_02, ..., F02_09]
    """
    features = []

    dif_val = dif[idx]
    dea_val = dea[idx]
    macd_val = (dif_val - dea_val) * 2

    # === F02_01~03: MACD归一化特征 [P0修复] ===
    # 修复：添加范围限制，假设MACD通常不超过close的10%
    dif_norm = safe_divide(dif_val, close, 0)
    dif_norm = max(-0.1, min(dif_norm, 0.1)) if np.isfinite(dif_norm) else 0.0
    features.append(dif_norm)  # F02_01: DIF归一化

    dea_norm = safe_divide(dea_val, close, 0)
    dea_norm = max(-0.1, min(dea_norm, 0.1)) if np.isfinite(dea_norm) else 0.0
    features.append(dea_norm)  # F02_02: DEA归一化

    macd_norm = safe_divide(macd_val, close, 0)
    macd_norm = max(-0.1, min(macd_norm, 0.1)) if np.isfinite(macd_norm) else 0.0
    features.append(macd_norm)  # F02_03: MACD柱归一化

    # === F02_04/05: MACD交叉信号 ===
    if idx > 0:
        is_golden = dif_val > dea_val and dif[idx-1] <= dea[idx-1]
        is_dead = dif_val < dea_val and dif[idx-1] >= dea[idx-1]
    else:
        is_golden = is_dead = False
    features.append(1.0 if is_golden else 0.0)  # F02_04: MACD金叉
    features.append(1.0 if is_dead else 0.0)  # F02_05: MACD死叉

    # === F02_06~07: MACD变化率 [P0修复] ===
    # 修复：添加范围限制到±5%，避免极端变化率主导模型
    if idx > 0:
        dif_change = safe_divide(dif_val - dif[idx-1], close, 0)
        dif_change = max(-0.05, min(dif_change, 0.05)) if np.isfinite(dif_change) else 0.0
        dea_change = safe_divide(dea_val - dea[idx-1], close, 0)
        dea_change = max(-0.05, min(dea_change, 0.05)) if np.isfinite(dea_change) else 0.0
    else:
        dif_change = dea_change = 0
    features.append(dif_change)  # F02_06: DIF变化率
    features.append(dea_change)  # F02_07: DEA变化率

    # === F02_08: DIF-DEA发散速度 ★P0纠缠态 ===
    # (改进：指数加权平均替代固定5根窗口)
    # ✅ Bug#18修复：历史数组不应包含当前值
    dif_dea_cohesion = safe_divide(abs(dif_val - dea_val), close, 0)
    if idx >= 10:
        # ✅ 向量化：使用NumPy数组切片替代Python循环
        look_back_start = max(0, idx - 10)
        look_back_end = idx

        # 提取历史数据段 [idx-10, idx-1]
        hist_indices = np.arange(look_back_start, look_back_end)
        hist_dif = dif[hist_indices]
        hist_dea = dea[hist_indices]
        hist_closes = closes[hist_indices]

        # 向量化计算：close_safe处理NaN和EPS
        closes_valid = np.where(
            np.isfinite(hist_closes) & (hist_closes > EPS),
            hist_closes,
            EPS
        )

        # ✅ NaN防护：对DIF和DEA也做NaN检查
        hist_dif_safe = np.where(np.isfinite(hist_dif), hist_dif, 0.0)
        hist_dea_safe = np.where(np.isfinite(hist_dea), hist_dea, 0.0)

        # 向量化计算历史cohesion
        dif_dea_history = np.abs(hist_dif_safe - hist_dea_safe) / closes_valid

        if len(dif_dea_history) > 1:
            dif_dea_divergence_speed = compute_exponential_velocity(
                dif_dea_cohesion, dif_dea_history, half_life=5
            )
            # ✅ NaN防护：确保最终输出不是NaN
            dif_dea_divergence_speed = dif_dea_divergence_speed if np.isfinite(dif_dea_divergence_speed) else 0.0
        else:
            dif_dea_divergence_speed = 0
    else:
        dif_dea_divergence_speed = 0
    features.append(dif_dea_divergence_speed)  # F02_08

    # === F02_09: MACD柱加速度 ===
    # 定义：MACD柱值的二阶动量（变化率的变化率）
    # ✅ 审计修复：改用绝对变化量，避免分母符号问题
    if idx > 1:
        macd_now = macd_histogram[idx]
        macd_prev = macd_histogram[idx - 1]
        macd_prev_prev = macd_histogram[idx - 2]

        # ✅ NaN检查：如果有NaN，返回0
        if not (np.isfinite(macd_now) and np.isfinite(macd_prev) and np.isfinite(macd_prev_prev)):
            macd_acceleration = 0
        else:
            # 改用绝对变化量（放弃比例计算）
            macd_momentum_now = macd_now - macd_prev
            macd_momentum_prev = macd_prev - macd_prev_prev
            accel_raw = (macd_momentum_now - macd_momentum_prev) / 0.05
            # ✅ 修复NaN传播：np.clip会传播NaN，改用max/min
            macd_acceleration = max(-1.0, min(accel_raw, 1.0)) if np.isfinite(accel_raw) else 0.0
    else:
        macd_acceleration = 0
    features.append(macd_acceleration)  # F02_09

    return features


# 导出
__all__ = ['extract_f02_features']
