"""F03_02: MA60成交量波动"""
import numpy as np


def calculate(idx_array, ma60_volumes):
    """
    MA60 成交量的历史波动性
    
    Args:
        idx_array: (batch_size,) 样本索引
        ma60_volumes: (total_bars,) MA60成交量
    
    Returns:
        (batch_size,) 成交量波动特征 [0, 2]
    """
    batch_size = len(idx_array)
    features = np.ones(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        try:
            if idx >= 60:
                ma60_hist = ma60_volumes[idx - 60:idx + 1]
            else:
                ma60_hist = ma60_volumes[max(0, idx - 60):idx + 1]

            valid_mask = np.isfinite(ma60_hist)
            if np.sum(valid_mask) > 0:
                ma60_hist_mean = np.mean(ma60_hist[valid_mask])
            else:
                ma60_hist_mean = np.nan

            if np.isfinite(ma60_hist_mean) and ma60_hist_mean > 1e-8:
                ma60_vol_curr = ma60_volumes[idx]
                if np.isfinite(ma60_vol_curr):
                    ma60_vol_norm = np.clip(ma60_vol_curr / ma60_hist_mean, 0.0, 2.0)
                else:
                    ma60_vol_norm = 1.0
            else:
                ma60_vol_norm = 1.0

            features[i] = ma60_vol_norm if np.isfinite(ma60_vol_norm) else 1.0
        except:
            features[i] = 1.0
    
    return features
