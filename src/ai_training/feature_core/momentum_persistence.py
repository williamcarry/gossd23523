"""
动量持续性特征组（F08_01~F08_08）

本模块包含所有与动量持续性相关的8个特征计算
从 feature_extractor.py 提取，保持100%一致

【P0核心特征】动量持续性特征包括：
- F08_01~F08_05: 分板块相对强弱（5个独立特征）
- F08_06: 金叉持续天数
- F08_07: 趋势持续天数
- F08_08: 放量持续天数
"""
import numpy as np
from .config import EPS


def calculate_f08_01_sector_relative_strength(stock_code, stock_klines, market_index_klines_dict, idx, period=20, norm_range=0.20):
    """
    F08_01~F08_05: 分板块相对强弱（5个独立特征，根据股票代码自动激活对应板块）
    
    🔥 高盛标准改进版（Goldman Sachs Standard）：
    1. 使用对数收益率（可加性+正态分布更好）
    2. Beta系数调整（计算超额收益）
    3. 动态标准差归一化（自适应市场波动）
    
    返回5个特征值（对应F08_01~F08_05）：
        F08_01: 相对上证指数强弱（600/601/603/605开头股票有值，其他为0）
        F08_02: 相对深证成指强弱（000/001/003开头股票有值，其他为0）
        F08_03: 相对创业板指强弱（300开头股票有值，其他为0）
        F08_04: 相对科创50强弱（688开头股票有值，其他为0）
        F08_05: 相对北证50强弱（8xxxxx/43xxxx/82xxxx/83xxxx开头股票有值，其他为0）
    
    参数:
        stock_code: 股票代码（如'600000'），用于识别所属板块
        stock_klines: 股票K线数据列表
        market_index_klines_dict: 指数K线字典 {
            'sh.000001': [...],  # 上证指数K线
            'sz.399001': [...],  # 深证成指K线
            'sz.399006': [...],  # 创业板指K线
            'sh.000688': [...],  # 科创50K线
            'bj.899050': [...]   # 北证50K线
        }
        idx: 计算位置
        period: 计算周期（默认20日）
        norm_range: 归一化范围（默认±20%，仅在历史数据不足时使用）
    
    返回:
        tuple: (F08_01, F08_02, F08_03, F08_04, F08_05) - 5个相对强弱特征值
    """
    # 初始化5个特征为0
    f08_01_sh000001 = 0.0  # 上证
    f08_01_sz399001 = 0.0  # 深证
    f08_01_sz399006 = 0.0  # 创业板
    f08_01_sh000688 = 0.0  # 科创50
    f08_01_bj899050 = 0.0  # 北证50
    
    # 边界检查
    if not stock_code or len(stock_klines) < period:
        return f08_01_sh000001, f08_01_sz399001, f08_01_sz399006, f08_01_sh000688, f08_01_bj899050
    
    # 如果没有提供指数数据，全部返回0
    if not market_index_klines_dict:
        return f08_01_sh000001, f08_01_sz399001, f08_01_sz399006, f08_01_sh000688, f08_01_bj899050
    
    # 识别股票所属板块
    index_code = None
    feature_position = None  # 1=上证, 2=深证, 3=创业板, 4=科创50, 5=北证50
    
    if stock_code.startswith(('600', '601', '603', '605')):
        index_code = 'sh.000001'  # 上证指数（沪市主板）
        feature_position = 1
    elif stock_code.startswith(('000', '001', '002', '003')):
        index_code = 'sz.399001'  # 深证成指（深市主板，含原中小板）
        feature_position = 2
    elif stock_code.startswith(('300', '301', '302')):
        index_code = 'sz.399006'  # 创业板指（创业板）
        feature_position = 3
    elif stock_code.startswith('688'):
        index_code = 'sh.000688'  # 科创50（科创板）
        feature_position = 4
    elif stock_code.startswith(('43', '82', '83', '87', '88', '89')) and len(stock_code) == 6:
        index_code = 'bj.899050'  # 北证50（北交所）
        feature_position = 5
    else:
        # 无法识别板块，全部返回0（表示与大盘不相关）
        return f08_01_sh000001, f08_01_sz399001, f08_01_sz399006, f08_01_sh000688, f08_01_bj899050
    
    # 获取对应的指数K线数据
    sector_klines = market_index_klines_dict.get(index_code)
    if not sector_klines or len(sector_klines) < period:
        # 没有对应指数数据，返回0
        return f08_01_sh000001, f08_01_sz399001, f08_01_sz399006, f08_01_sh000688, f08_01_bj899050
    
    try:
        # 获取股票收盘价
        stock_close_start = float(stock_klines[idx - period]['close'])
        stock_close_end = float(stock_klines[idx]['close'])
        
        # 获取板块指数收盘价
        sector_close_start = float(sector_klines[idx - period]['close'])
        sector_close_end = float(sector_klines[idx]['close'])
        
        # 检查有效性
        if stock_close_start <= 0 or sector_close_start <= 0 or stock_close_end <= 0 or sector_close_end <= 0:
            return f08_01_sh000001, f08_01_sz399001, f08_01_sz399006, f08_01_sh000688, f08_01_bj899050
        
        # ✅ 高盛标准1：使用对数收益率（log returns）
        # 优势：可加性、与正态分布更接近、极端波动更稳健
        stock_return = np.log(stock_close_end / stock_close_start)
        sector_return = np.log(sector_close_end / sector_close_start)
        
        # ✅ P1-F7修复（高盛标准）：Winsorize极端收益率，避免异常值影响
        # 金融标准：单日收益率>50%或<-50%为异常值（涨跌停板/黑天鹅事件）
        MAX_DAILY_RETURN = 0.50  # 50%
        if abs(stock_return) > MAX_DAILY_RETURN:
            stock_return = np.sign(stock_return) * MAX_DAILY_RETURN
        if abs(sector_return) > MAX_DAILY_RETURN:
            sector_return = np.sign(sector_return) * MAX_DAILY_RETURN
        
        # ✅ 高盛标准2：计算Beta系数（使用最近20个交易日）
        # Beta = Cov(stock, sector) / Var(sector)
        beta = 1.0  # 默认Beta=1
        if idx >= period + 20:  # 需要至少40个数据点（20+20）才计算Beta
            try:
                # 向量化计算：提取最近20个交易日的收盘价
                start_idx = max(idx - 20, 1)
                end_idx = idx

                # 提取历史数据段 [start_idx, end_idx]
                s_closes_prev = np.array([float(stock_klines[i - 1]['close']) for i in range(start_idx, end_idx)])
                s_closes_curr = np.array([float(stock_klines[i]['close']) for i in range(start_idx, end_idx)])
                sec_closes_prev = np.array([float(sector_klines[i - 1]['close']) for i in range(start_idx, end_idx)])
                sec_closes_curr = np.array([float(sector_klines[i]['close']) for i in range(start_idx, end_idx)])

                # 向量化计算：检查有效性并计算日收益率
                valid_mask = (s_closes_prev > 0) & (s_closes_curr > 0) & (sec_closes_prev > 0) & (sec_closes_curr > 0)

                if np.sum(valid_mask) > 0:
                    # 向量化计算对数收益率
                    s_ret = np.log(s_closes_curr[valid_mask] / s_closes_prev[valid_mask])
                    sec_ret = np.log(sec_closes_curr[valid_mask] / sec_closes_prev[valid_mask])

                    # 向量化Winsorize：处理极端值
                    # ✅ 修复NaN传播：np.clip会传播NaN，改用np.where过滤
                    s_ret = np.where(np.isfinite(s_ret), np.clip(s_ret, -MAX_DAILY_RETURN, MAX_DAILY_RETURN), 0.0)
                    sec_ret = np.where(np.isfinite(sec_ret), np.clip(sec_ret, -MAX_DAILY_RETURN, MAX_DAILY_RETURN), 0.0)

                    # 向量化有效性检查
                    finite_mask = np.isfinite(s_ret) & np.isfinite(sec_ret)
                    stock_returns_hist = s_ret[finite_mask]
                    sector_returns_hist = sec_ret[finite_mask]

                    # 计算Beta系数
                    # ✅ P0修复：高盛标准要求至少20个样本，确保Beta可靠性
                    if len(stock_returns_hist) >= 20:  # ✅ 从10改为20
                        # ✅ NaN防护：np.cov可能返回NaN，需要先检查有效性
                        try:
                            cov_matrix = np.cov(stock_returns_hist, sector_returns_hist)
                            if np.isfinite(cov_matrix[0, 1]):
                                covariance = cov_matrix[0, 1]
                            else:
                                covariance = 0.0

                            sector_variance = np.var(sector_returns_hist)
                            if not np.isfinite(sector_variance) or sector_variance <= 0:
                                sector_variance = 0.0
                        except:
                            covariance = 0.0
                            sector_variance = 0.0

                        if sector_variance > 1e-8:
                            beta = covariance / sector_variance
                            # Beta范围限制在[0.3, 3.0]（防止异常值）
                            # ✅ 修复NaN传播：np.clip会传播NaN，改用max/min
                            beta = max(0.3, min(beta, 3.0)) if np.isfinite(beta) else 1.0
                        else:
                            beta = 1.0
            except:
                beta = 1.0
        
        # ✅ 高盛标准3：计算超额收益（Excess Return）
        # 超额收益 = 股票收益 - Beta * 板块收益
        excess_return = stock_return - beta * sector_return
        
        # ✅ 高盛标准4：动态标准差归一化
        # 使用历史相对收益的标准差作为归一化基准
        relative_std = norm_range  # 默认使用固定值
        if idx >= period + 20 and 'stock_returns_hist' in locals() and len(stock_returns_hist) >= 20:  # ✅ 与Beta样本数保持一致
            try:
                # 向量化计算：历史相对收益（超额收益）
                relative_returns_hist = stock_returns_hist - beta * sector_returns_hist
                # ✅ NaN防护：np.std可能返回NaN，需要先检查有效性
                if np.all(np.isfinite(relative_returns_hist)):
                    relative_std = np.std(relative_returns_hist)
                    if np.isfinite(relative_std) and relative_std > 0:
                        # ✅ P1修复：最小值从0.01降至0.003（0.3%），避免市场平静期过度敏感
                        relative_std = max(relative_std, 0.003)  # ✅ 从0.01改为0.003
                    else:
                        relative_std = norm_range
                else:
                    relative_std = norm_range
            except:
                relative_std = norm_range
        
        # ✅ NaN防护：确保计算过程中没有NaN
        if not (np.isfinite(excess_return) and np.isfinite(relative_std) and relative_std > 0):
            return f08_01_sh000001, f08_01_sz399001, f08_01_sz399006, f08_01_sh000688, f08_01_bj899050

        # 归一化相对强弱
        relative_strength = excess_return / relative_std
        # ✅ NaN防护：最终输出检查
        if not np.isfinite(relative_strength):
            relative_strength = 0.0
        relative_strength = max(-1.0, min(1.0, relative_strength))
        
        # 只激活对应板块的特征
        if feature_position == 1:
            f08_01_sh000001 = relative_strength
        elif feature_position == 2:
            f08_01_sz399001 = relative_strength
        elif feature_position == 3:
            f08_01_sz399006 = relative_strength
        elif feature_position == 4:
            f08_01_sh000688 = relative_strength
        elif feature_position == 5:
            f08_01_bj899050 = relative_strength
        
        return f08_01_sh000001, f08_01_sz399001, f08_01_sz399006, f08_01_sh000688, f08_01_bj899050
        
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return f08_01_sh000001, f08_01_sz399001, f08_01_sz399006, f08_01_sh000688, f08_01_bj899050


def calculate_f08_06_golden_cross_duration(dif, dea, idx, max_days=20):
    """
    F08_06: 金叉持续天数

    参数:
        dif: MACD DIF数组
        dea: MACD DEA数组
        idx: 计算位置
        max_days: 最大天数（默认20天）

    返回:
        float: 金叉持续天数 [0, 1]
    """
    # 边界检查
    if idx < 1 or idx >= len(dif):
        return 0.0

    try:
        # 检查当前是否金叉状态（DIF > DEA）
        if dif[idx] <= dea[idx]:
            return 0.0

        # 向量化计算：回溯找金叉起始点
        max_lookback = min(idx + 1, max_days + 1)  # 包含当前点
        start_idx = idx - max_lookback + 1
        if start_idx < 0:
            start_idx = 0

        # 提取查找区间的数据
        lookup_indices = np.arange(start_idx, idx + 1)
        dif_subset = dif[lookup_indices]
        dea_subset = dea[lookup_indices]

        # 检查有效性和金叉条件
        valid_mask = np.isfinite(dif_subset) & np.isfinite(dea_subset)
        golden_mask = valid_mask & (dif_subset > dea_subset)

        # 从后往前找连续的金叉点（从idx向后回溯）
        # 逆序检查，找到第一个非金叉点之前的连续金叉数
        days_count = 0
        for i in range(len(lookup_indices) - 1, -1, -1):
            if golden_mask[i]:
                days_count += 1
            else:
                break

        # 归一化到 [0, 1]
        return min(days_count / max_days, 1.0)

    except (IndexError, ValueError, TypeError):
        return 0.0


def calculate_f08_07_trend_duration(ma25, idx, max_days=30):
    """
    F08_07: 趋势持续天数

    参数:
        ma25: MA25均线数组
        idx: 计算位置
        max_days: 最大天数（默认30天）

    返回:
        float: 趋势持续天数 [-1, 1]
    """
    # 边界检查
    if idx < 1 or idx >= len(ma25):
        return 0.0

    try:
        # 判断当前方向
        if ma25[idx] > ma25[idx - 1]:
            direction = 1  # 向上
        elif ma25[idx] < ma25[idx - 1]:
            direction = -1  # 向下
        else:
            return 0.0  # 横盘

        # 向量化计算：回溯找趋势连续点数
        max_lookback = min(idx, max_days)
        start_idx = idx - max_lookback
        if start_idx < 1:
            start_idx = 1

        # 提取查找区间的数据 [start_idx-1, idx]
        lookup_indices = np.arange(start_idx - 1, idx + 1)
        ma25_subset = ma25[lookup_indices]

        # 向量化计算差分（当前-前一个）
        ma25_diffs = ma25_subset[1:] - ma25_subset[:-1]

        # 检查有效性
        valid_mask = np.isfinite(ma25_diffs)

        # 根据方向确定趋势mask
        if direction > 0:
            trend_mask = valid_mask & (ma25_diffs > 0)
        else:
            trend_mask = valid_mask & (ma25_diffs < 0)

        # 从后往前计数连续的趋势点
        days_count = 0
        for i in range(len(trend_mask) - 1, -1, -1):
            if trend_mask[i]:
                days_count += 1
            else:
                break

        # 加上当前这一根
        days_count += 1

        # 归一化到 [-1, 1]
        trend_strength = direction * min(days_count / max_days, 1.0)
        return trend_strength

    except (IndexError, ValueError, TypeError):
        return 0.0


def calculate_f08_08_volume_duration(volumes, idx, period=20, multiplier=1.5, max_days=10):
    """
    F08_08: 放量持续天数

    参数:
        volumes: 成交量数组
        idx: 计算位置
        period: 基准计算周期（默认20日）
        multiplier: 放量倍数（默认1.5倍）
        max_days: 最大天数（默认10天）

    返回:
        float: 放量持续天数 [0, 1]
    """
    # 边界检查
    if idx < period or idx >= len(volumes):
        return 0.0

    try:
        # 向量化计算：基准成交量（过去20日平均，不包含当前）
        baseline_volumes = volumes[idx - period:idx]

        # 向量化过滤无效值
        valid_volumes = baseline_volumes[np.isfinite(baseline_volumes)]
        if len(valid_volumes) < period * 0.5:  # 至少需要50%的有效数据
            return 0.0

        avg_volume = np.mean(valid_volumes)

        # 放量阈值
        threshold = avg_volume * multiplier

        # 检查当前是否放量
        if not np.isfinite(volumes[idx]) or volumes[idx] <= threshold:
            return 0.0

        # 向量化计算：回溯连续放量天数
        max_lookback = min(idx + 1, max_days + 1)
        start_idx = idx - max_lookback + 1
        if start_idx < 0:
            start_idx = 0

        # 提取查找区间的数据
        lookup_indices = np.arange(start_idx, idx + 1)
        volume_subset = volumes[lookup_indices]

        # 向量化检查：有效性 & 超过阈值
        valid_mask = np.isfinite(volume_subset) & (volume_subset > threshold)

        # 从后往前计数连续的放量点
        days_count = 0
        for i in range(len(valid_mask) - 1, -1, -1):
            if valid_mask[i]:
                days_count += 1
            else:
                break

        # 归一化到 [0, 1]
        return min(days_count / max_days, 1.0)

    except (IndexError, ValueError, TypeError):
        return 0.0


def extract_f08_features(
    idx, closes, volumes,
    ma25_prices, dif, dea,
    stock_code=None,
    market_index_klines_dict=None
):
    """
    提取动量持续性特征组（F08_01~F08_08）
    
    参数:
        idx: 当前K线索引
        closes: 收盘价数组
        volumes: 成交量数组
        ma25_prices: MA25数组
        dif: MACD DIF数组
        dea: MACD DEA数组
        stock_code: 股票代码（可选，用于F08_01~F08_05）
        market_index_klines_dict: 指数K线字典（可选，用于F08_01~F08_05）
    
    返回:
        list: 包含8个特征值的列表 [F08_01, F08_02, F08_03, F08_04, F08_05, F08_06, F08_07, F08_08]
    """
    features = []
    
    # === F08_01~F08_05: 分板块相对强弱（5个特征） ===
    # 根据股票代码自动激活对应板块的特征，其他板块为0
    try:
        # 准备股票K线数据（字典格式）
        stock_klines_dict = [{'close': closes[j]} for j in range(len(closes))]
        
        # 准备指数K线数据字典（字典格式）
        index_klines_dict_formatted = {}
        if market_index_klines_dict:
            for index_code, index_klines in market_index_klines_dict.items():
                index_klines_dict_formatted[index_code] = [{'close': float(k.close)} for k in index_klines]
        
        # 调用新版F08_01函数，返回5个特征
        f08_01_sh000001, f08_01_sz399001, f08_01_sz399006, f08_01_sh000688, f08_01_bj899050 = \
            calculate_f08_01_sector_relative_strength(
                stock_code, 
                stock_klines_dict, 
                index_klines_dict_formatted, 
                idx
            )
    except Exception as e:
        # 异常时全部返回0
        f08_01_sh000001 = f08_01_sz399001 = f08_01_sz399006 = f08_01_sh000688 = f08_01_bj899050 = 0.0
    
    features.append(f08_01_sh000001)  # F08_01: 相对上证指数强弱
    features.append(f08_01_sz399001)  # F08_02: 相对深证成指强弱
    features.append(f08_01_sz399006)  # F08_03: 相对创业板指强弱
    features.append(f08_01_sh000688)  # F08_04: 相对科创50强弱
    features.append(f08_01_bj899050)  # F08_05: 相对北证50强弱
    
    # === F08_06: 金叉持续天数 ===
    try:
        f08_06 = calculate_f08_06_golden_cross_duration(dif, dea, idx)
    except Exception as e:
        f08_06 = 0.0
    features.append(f08_06)  # F08_06
    
    # === F08_07: 趋势持续天数 ===
    try:
        f08_07 = calculate_f08_07_trend_duration(ma25_prices, idx)
    except Exception as e:
        f08_07 = 0.0
    features.append(f08_07)  # F08_07
    
    # === F08_08: 放量持续天数 ===
    try:
        f08_08 = calculate_f08_08_volume_duration(volumes, idx)
    except Exception as e:
        f08_08 = 0.0
    features.append(f08_08)  # F08_08
    
    return features


# 导出
__all__ = [
    'extract_f08_features',
    'calculate_f08_01_sector_relative_strength',
    'calculate_f08_06_golden_cross_duration',
    'calculate_f08_07_trend_duration',
    'calculate_f08_08_volume_duration',
]
