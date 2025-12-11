"""
支撑阻力特征组（F06_01）

本模块包含支撑阻力相关的1个特征计算
从 feature_extractor.py 提取，保持100%一致
"""
import numpy as np
from .config import EPS
from .utils import compute_capital_persistence


def extract_f06_features(
    idx, volumes, ma60_volumes
):
    """
    提取支撑阻力特征组（F06_01）
    
    参数:
        idx: 当前K线索引
        volumes: 成交量数组
        ma60_volumes: MA60成交量数组
    
    返回:
        list: 包含1个特征值的列表 [F06_01]
    """
    features = []
    
    # === F06_01: 资金持续关注度 ===
    # (替代：成交量斜率与方向配合) - 最近N根中有多少根持续放量
    # ✅ P0修复：传入日成交量数组，而非MA5量线数组
    if idx >= 5:
        # 修复：安全获取baseline，避免NaN风险
        # ✅ P2修复：检查有限性（同时检查NaN和Inf）
        # ✅ 向量化：直接检查当前值，无需转换
        current_ma60 = ma60_volumes[idx]
        if np.isfinite(current_ma60):
            baseline_vol = current_ma60
        else:
            # 向量化计算：提取有效的MA60值并计算平均
            valid_range = ma60_volumes[max(0, idx-59):idx+1]
            valid_vals = valid_range[np.isfinite(valid_range)]
            baseline_vol = np.mean(valid_vals) if len(valid_vals) > 0 else 1.0

        # 向量化：直接使用numpy数组，避免list转换
        vol_window = volumes[max(0, idx-4):idx+1]

        # ✅ NaN防护：确保baseline_vol有效
        if not np.isfinite(baseline_vol) or baseline_vol <= 0:
            baseline_vol = 1.0

        try:
            capital_persistence = compute_capital_persistence(
                vol_window,  # ✅ 修复：传入日成交量数组（numpy array或list均可）
                baseline_vol,
                window=5
            )
            # ✅ NaN防护：检查返回值的有效性
            if not np.isfinite(capital_persistence):
                capital_persistence = 0.5
        except:
            capital_persistence = 0.5
    else:
        capital_persistence = 0.5
    features.append(capital_persistence)  # F06_01
    
    return features


# 导出
__all__ = ['extract_f06_features']
