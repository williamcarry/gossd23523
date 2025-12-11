"""
特征核心模块 - 全局配置

本文件包含所有特征计算使用的全局常量和配置类
从 feature_extractor.py 提取，保持100%一致
"""
import numpy as np

# ============================================================================
# 全局常数
# ============================================================================
EPS = 1e-8
DEFAULT_WHEN_ZERO = 0
VOL_CROSS_THRESHOLD = 0.05  # 量能金叉阈值5%


# ============================================================================
# 特征归一化配置
# ============================================================================
class FeatureNormConfig:
    """特征归一化配置类"""
    
    # 范围限制参数
    MACD_RANGE_LIMIT = 0.1
    MACD_CHANGE_LIMIT = 0.05
    VOLUME_RATIO_MAX = 3.0
    PRICE_MA_STRENGTH_LIMIT = 1.0
    COHESION_MAX = 1.0
    
    # 形态识别阈值
    DOJI_BODY_THRESHOLD_STRONG = 0.1
    DOJI_BODY_THRESHOLD_WEAK = 0.2
    UPPER_SHADOW_RATIO = 1.5
    BODY_RATIO_MAX = 2.0
    
    # 成交量相关阈值
    VOLUME_BASELINE_MULTIPLIER = 1.2
    VOLUME_EXTREME_MULTIPLIER = 1.5
    
    # 融合权重系数
    ATR_LEVEL_WEIGHT = 0.3
    ATR_MOMENTUM_WEIGHT = 0.7
    CAPITAL_PERSIST_WEIGHT = 0.6
    CAPITAL_TREND_WEIGHT = 0.4
    
    # 归一化转换参数
    MA_NORM_OFFSET = 0.5
    MA_NORM_SCALE = 2.0
    ACCELERATION_SCALE = 0.05
    DISTANCE_PERCENT_SCALE = 3.0
    DISTANCE_PRICE_SCALE = 5.0
    
    # 百分位阈值
    PERCENTILE_10 = 10
    PERCENTILE_25 = 25
    PERCENTILE_75 = 75
    PERCENTILE_90 = 90
    
    # 时间窗口参数
    SIGNAL_AGE_DECAY_FACTOR = 3.0


# 配置实例（全局单例）
CFG = FeatureNormConfig()


# ============================================================================
# 导出
# ============================================================================
__all__ = [
    'EPS',
    'DEFAULT_WHEN_ZERO',
    'VOL_CROSS_THRESHOLD',
    'FeatureNormConfig',
    'CFG',
]
