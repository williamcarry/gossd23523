"""F02_04: MACD金叉信号"""
import numpy as np


def calculate(idx_array, dif, dea):
    """
    MACD DIF向上穿越DEA的金叉信号
    
    Args:
        idx_array: (batch_size,) 样本索引
        dif: (total_bars,) DIF值
        dea: (total_bars,) DEA值
    
    Returns:
        (batch_size,) 金叉信号 (1.0=是，0.0=否)
    """
    batch_size = len(idx_array)
    features = np.zeros(batch_size, dtype=np.float32)
    
    for i, idx in enumerate(idx_array):
        if idx > 0:
            dif_curr = dif[idx]
            dea_curr = dea[idx]
            dif_prev = dif[idx-1]
            dea_prev = dea[idx-1]
            
            if (np.isfinite(dif_curr) and np.isfinite(dea_curr) and 
                np.isfinite(dif_prev) and np.isfinite(dea_prev)):
                is_golden = (dif_curr > dea_curr) and (dif_prev <= dea_prev)
                features[i] = 1.0 if is_golden else 0.0
            else:
                features[i] = 0.0
        else:
            features[i] = 0.0
    
    return features
