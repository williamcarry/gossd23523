"""F08_06: MACD金叉持续天数"""
import numpy as np


def calculate(idx_array, dif, dea):
    """
    MACD保持金叉状态的持续天数
    
    Args:
        idx_array: (batch_size,) 样本索引
        dif: (total_bars,) DIF值
        dea: (total_bars,) DEA值
    
    Returns:
        (batch_size,) 持续天数 [0, 255]（255表示>255）
    """
    batch_size = len(idx_array)
    features = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        consecutive_days = 0
        for j in range(idx, max(-1, idx - 255), -1):
            if j >= 0 and np.isfinite(dif[j]) and np.isfinite(dea[j]):
                if dif[j] > dea[j]:
                    consecutive_days += 1
                else:
                    break
            else:
                break
        
        features[i] = min(consecutive_days, 255)
    
    return features
