"""F08_01: 上证指数相对强弱"""
import numpy as np


def calculate(batch_size):
    """
    动量持续性: 上证指数相对强弱
    
    需要大盘指数数据（沪深300、上证指数等），暂简化为零值
    
    Args:
        batch_size: 样本数量
    
    Returns:
        (batch_size,) 特征值（目前为零）
    """
    return np.zeros(batch_size, dtype=np.float32)
