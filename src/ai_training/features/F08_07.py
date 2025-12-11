"""F08_07: 趋势向上持续天数"""
import numpy as np


def calculate(idx_array, closes):
    """
    收盘价保持上升趋势的持续天数
    
    Args:
        idx_array: (batch_size,) 样本索引
        closes: (total_bars,) 收盘价
    
    Returns:
        (batch_size,) 持续天数 [0, 255]
    """
    batch_size = len(idx_array)
    features = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        consecutive_days = 0
        for j in range(idx, max(-1, idx - 255), -1):
            if j >= 1 and np.isfinite(closes[j]) and np.isfinite(closes[j-1]):
                if closes[j] > closes[j-1]:
                    consecutive_days += 1
                else:
                    break
            else:
                break
        
        features[i] = min(consecutive_days, 255)
    
    return features
