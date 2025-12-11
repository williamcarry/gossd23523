"""F02_08: DIF-DEA发散速度"""
import numpy as np
from .utils import safe_divide_batch


def calculate(idx_array, dif, dea, closes):
    """
    DIF与DEA距离的历史变化速度
    
    Args:
        idx_array: (batch_size,) 样本索引
        dif: (total_bars,) DIF值
        dea: (total_bars,) DEA值
        closes: (total_bars,) 收盘价
    
    Returns:
        (batch_size,) 发散速度
    """
    batch_size = len(idx_array)
    features = np.zeros(batch_size, dtype=np.float32)
    
    # 计算当前DIF-DEA距离
    dif_dea_dist = safe_divide_batch(np.abs(dif - dea), closes, 0.0)
    
    for i, idx in enumerate(idx_array):
        if idx >= 10:
            try:
                hist_dif_dea = np.abs(dif[idx-10:idx] - dea[idx-10:idx])
                hist_close = closes[idx-10:idx]
                hist_dist = safe_divide_batch(hist_dif_dea, hist_close, 0.0)
                
                valid_hist_mask = np.isfinite(hist_dist)
                if np.sum(valid_hist_mask) > 0:
                    valid_hist_dist = hist_dist[valid_hist_mask]
                    weights = np.exp(-0.2 * np.arange(len(valid_hist_dist))[::-1])
                    weights /= weights.sum()
                    divergence_speed = np.sum(weights * np.minimum(valid_hist_dist, 1.0))
                else:
                    divergence_speed = 0.0
                
                result = dif_dea_dist[idx] - divergence_speed
                features[i] = result if np.isfinite(result) else 0.0
            except Exception:
                features[i] = 0.0
        else:
            features[i] = dif_dea_dist[idx] if np.isfinite(dif_dea_dist[idx]) else 0.0
    
    return features
