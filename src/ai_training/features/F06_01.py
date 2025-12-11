"""F06_01: 资金持续关注度"""
import numpy as np


def calculate(idx_array, volumes, ma60_volumes):
    """
    最近5根K线中有多少根持续放量
    
    Args:
        idx_array: (batch_size,) 样本索引数组
        volumes: (total_bars,) 成交量数组
        ma60_volumes: (total_bars,) MA60成交量
    
    Returns:
        (batch_size,) 资金持续关注度 [0, 1]
    """
    batch_size = len(idx_array)
    capital_persistence = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        if idx >= 5:
            current_ma60 = ma60_volumes[idx]
            if np.isfinite(current_ma60):
                baseline_vol = current_ma60
            else:
                valid_range = ma60_volumes[max(0, idx - 59):idx + 1]
                valid_vals = valid_range[np.isfinite(valid_range)]
                baseline_vol = np.mean(valid_vals) if len(valid_vals) > 0 else 1.0

            if not np.isfinite(baseline_vol) or baseline_vol <= 0:
                baseline_vol = 1.0

            try:
                vol_window = volumes[max(0, idx - 4):idx + 1]
                count_volume_above = np.sum(vol_window > baseline_vol)
                persist = count_volume_above / 5.0
                persist = np.clip(persist, 0.0, 1.0) if np.isfinite(persist) else 0.5
            except:
                persist = 0.5
        else:
            persist = 0.5
        capital_persistence[i] = persist
    
    return capital_persistence
