"""F03_03: 成交量持续性特征1"""
import numpy as np


def calculate(batch_size):
    """
    成交量持续性特征1（简化实现）
    
    Args:
        batch_size: 样本数量
    
    Returns:
        (batch_size,) 特征值（目前为零）
    """
    return np.zeros(batch_size, dtype=np.float32)
