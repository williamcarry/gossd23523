"""F08_08: 持续放量天数"""
import numpy as np


def calculate(idx_array, volumes, ma60_volumes):
    """
    成交量持续高于MA60的天数
    
    Args:
        idx_array: (batch_size,) 样本索引
        volumes: (total_bars,) 成交量
        ma60_volumes: (total_bars,) MA60成交量
    
    Returns:
        (batch_size,) 持续天数 [0, 255]
    """
    batch_size = len(idx_array)
    features = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        consecutive_days = 0
        for j in range(idx, max(-1, idx - 255), -1):
            if j >= 0 and np.isfinite(volumes[j]) and np.isfinite(ma60_volumes[j]):
                if volumes[j] > ma60_volumes[j]:
                    consecutive_days += 1
                else:
                    break
            else:
                break
        
        features[i] = min(consecutive_days, 255)
    
    return features
