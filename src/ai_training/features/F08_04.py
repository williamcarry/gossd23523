"""F08_04: 科创50相对强弱"""
import numpy as np


def calculate(batch_size):
    """
    动量特征: 科创50相对强弱
    
    Args:
        batch_size: 样本数量
    
    Returns:
        (batch_size,) 特征值（目前为零）
    """
    return np.zeros(batch_size, dtype=np.float32)
