"""
特征核心模块 - 按功能域分组设计

本模块将51个特征按功能域分组，便于维护、测试和复用

分组逻辑：
1. price_ma_features      - 价格均线特征（F01，13个）
2. macd_features          - MACD指标特征（F02，9个）
3. volume_features        - 成交量特征（F03，12个）
4. volatility_features    - 波动率特征（F04，3个）
5. trend_features         - 趋势特征（F05，2个）
6. support_resistance     - 支撑阻力特征（F06，1个）
7. strategy_2560          - 2560战法特征（F07，3个）
8. momentum_persistence   - 动量持续性特征（F08，8个）

共计：51个特征
"""

__version__ = '1.0.0'

# 导出配置和工具
from .config import EPS, DEFAULT_WHEN_ZERO, VOL_CROSS_THRESHOLD, FeatureNormConfig, CFG
from .utils import *

# 导出特征分组模块
from .price_ma_features import extract_f01_features
from .macd_features import extract_f02_features
from .volume_features import extract_f03_features
from .volatility_features import extract_f04_features
from .trend_features import extract_f05_features
from .support_resistance import extract_f06_features
from .strategy_2560 import extract_f07_features
from .momentum_persistence import extract_f08_features


def extract_all_features(
    idx,
    # 价格数据
    close, open_price, high, low,
    opens, closes, highs, lows,
    # 均线数据
    ma5_prices, ma25_prices,
    ma5_volumes, ma60_volumes, volumes,
    # MACD数据
    dif, dea, macd_histogram,
    # 波动率数据
    atr, upper_bb, middle_bb, lower_bb,
    # 其他
    capital_persistence=0.5,
    stock_code=None,
    market_index_klines_dict=None
):
    """
    批量提取所有51个特征
    
    参数:
        idx: 当前K线索引
        close: 当前收盘价
        open_price: 当前开盘价
        high: 当前最高价
        low: 当前最低价
        opens: 开盘价数组
        closes: 收盘价数组
        highs: 最高价数组
        lows: 最低价数组
        ma5_prices: MA5价格数组
        ma25_prices: MA25价格数组
        ma5_volumes: MA5成交量数组
        ma60_volumes: MA60成交量数组
        volumes: 成交量数组
        dif: MACD DIF数组
        dea: MACD DEA数组
        macd_histogram: MACD柱数组
        atr: ATR数组
        upper_bb: 布林带上轨数组
        middle_bb: 布林带中轨数组
        lower_bb: 布林带下轨数组
        capital_persistence: 资金持续关注度（F06_01）
        stock_code: 股票代码（可选，用于F08_01~F08_05）
        market_index_klines_dict: 指数K线字典（可选，用于F08_01~F08_05）
    
    返回:
        list: 包含51个特征值的列表
    """
    features = []
    
    # F01: 价格均线特征（13个）
    features.extend(extract_f01_features(
        idx, close, open_price, high, low,
        opens, closes, highs, lows,
        ma5_prices, ma25_prices, atr
    ))
    
    # F02: MACD特征（9个）
    features.extend(extract_f02_features(
        idx, close, closes,
        dif, dea, macd_histogram
    ))
    
    # F03: 成交量特征（12个）
    features.extend(extract_f03_features(
        idx, volumes[idx], volumes, closes,
        ma5_volumes, ma60_volumes
    ))
    
    # F04: 波动率特征（3个）
    features.extend(extract_f04_features(
        idx, close,
        atr, upper_bb, middle_bb, lower_bb
    ))
    
    # F05: 趋势特征（2个）
    features.extend(extract_f05_features(
        idx, open_price, closes,
        atr, ma25_prices
    ))
    
    # F06: 支撑阻力特征（1个）
    features.extend(extract_f06_features(
        idx, volumes, ma60_volumes
    ))
    
    # F07: 2560战法特征（3个）
    features.extend(extract_f07_features(
        idx, close, closes,
        ma5_prices, ma25_prices, atr,
        ma5_volumes, ma60_volumes, volumes,
        dif, dea,
        highs, lows,
        capital_persistence
    ))
    
    # F08: 动量持续性特征（8个）
    features.extend(extract_f08_features(
        idx, closes, volumes,
        ma25_prices, dif, dea,
        stock_code=stock_code,
        market_index_klines_dict=market_index_klines_dict
    ))
    
    return features


__all__ = [
    # 配置
    'EPS',
    'DEFAULT_WHEN_ZERO',
    'VOL_CROSS_THRESHOLD',
    'FeatureNormConfig',
    'CFG',
    
    # 基础计算函数
    'rolling_mean_aligned',
    'calculate_ema',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_rsi_standard',
    'calculate_kdj_standard',
    
    # 工具函数（从 utils 导出）
    'safe_divide',
    'compute_price_acceleration_correct',
    'compute_exponential_velocity',
    'compute_atr_with_momentum',
    'compute_hammer_strength',
    'compute_engulfing_strength',
    'compute_doji_strength',
    'compute_candle_pattern_combined',
    'compute_close_position',
    'compute_volume_percentile',
    'compute_price_volume_correlation',
    'compute_capital_persistence',
    'compute_volume_extreme_event',
    'compute_ma25_slope_atr_normalized_linregress',
    'compute_ma5_slope_atr_normalized_linregress',
    'compute_volume_ma_slope_linregress',
    'compute_volume_ratio_metrics',
    'calculate_atr_recent_60bars',
    
    # 特征提取函数
    'extract_f01_features',  # 价格均线特征（13个）
    'extract_f02_features',  # MACD特征（9个）
    'extract_f03_features',  # 成交量特征（12个）
    'extract_f04_features',  # 波动率特征（3个）
    'extract_f05_features',  # 趋势特征（2个）
    'extract_f06_features',  # 支撑阻力特征（1个）
    'extract_f07_features',  # 2560战法特征（3个）
    'extract_f08_features',  # 动量持续性特征（8个）
    
    # 批量提取函数
    'extract_all_features',
]
