#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标特征模块
包含常用的技术分析指标计算
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union
import warnings

# 尝试导入TA-Lib，如果不可用则使用替代实现
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib未安装，将使用替代实现。建议安装TA-Lib以获得更好的性能。")

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    计算相对强弱指标 (RSI)
    
    Args:
        prices: 价格序列
        period: RSI周期，默认14
    
    Returns:
        RSI值序列
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.RSI(prices.values, timeperiod=period), index=prices.index)
    else:
        # 替代实现
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def calculate_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    计算MACD指标
    
    Args:
        prices: 价格序列
        fast_period: 快线周期，默认12
        slow_period: 慢线周期，默认26
        signal_period: 信号线周期，默认9
    
    Returns:
        包含MACD、信号线、柱状图的DataFrame
    """
    if TALIB_AVAILABLE:
        macd, signal, hist = talib.MACD(prices.values, fastperiod=fast_period, 
                                       slowperiod=slow_period, signalperiod=signal_period)
        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal,
            'MACD_Histogram': hist
        }, index=prices.index)
    else:
        # 替代实现
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period).mean()
        hist = macd - signal
        
        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal,
            'MACD_Histogram': hist
        }, index=prices.index)

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
    """
    计算布林带
    
    Args:
        prices: 价格序列
        period: 移动平均周期，默认20
        std_dev: 标准差倍数，默认2
    
    Returns:
        包含上轨、中轨、下轨、宽度的DataFrame
    """
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, 
                                           nbdevup=std_dev, nbdevdn=std_dev)
        bb_width = (upper - lower) / middle
        bb_position = (prices - lower) / (upper - lower)
        
        return pd.DataFrame({
            'BB_Upper': upper,
            'BB_Middle': middle,
            'BB_Lower': lower,
            'BB_Width': bb_width,
            'BB_Position': bb_position
        }, index=prices.index)
    else:
        # 替代实现
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        bb_width = (upper - lower) / middle
        bb_position = (prices - lower) / (upper - lower)
        
        return pd.DataFrame({
            'BB_Upper': upper,
            'BB_Middle': middle,
            'BB_Lower': lower,
            'BB_Width': bb_width,
            'BB_Position': bb_position
        }, index=prices.index)

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    计算随机指标
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        k_period: %K周期，默认14
        d_period: %D周期，默认3
    
    Returns:
        包含%K和%D的DataFrame
    """
    if TALIB_AVAILABLE:
        slowk, slowd = talib.STOCH(high.values, low.values, close.values, 
                                   fastk_period=k_period, slowk_period=d_period, 
                                   slowd_period=d_period)
        return pd.DataFrame({
            'Stoch_K': slowk,
            'Stoch_D': slowd
        }, index=close.index)
    else:
        # 替代实现
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'Stoch_K': k,
            'Stoch_D': d
        }, index=close.index)

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算威廉指标 (%R)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期，默认14
    
    Returns:
        威廉指标序列
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.WILLR(high.values, low.values, close.values, timeperiod=period), 
                        index=close.index)
    else:
        # 替代实现
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算平均真实波幅 (ATR)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期，默认14
    
    Returns:
        ATR序列
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), 
                        index=close.index)
    else:
        # 替代实现
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    检测K线形态
    
    Args:
        df: 包含OHLC数据的DataFrame
    
    Returns:
        包含形态检测结果的DataFrame
    """
    result = df.copy()
    
    # 计算K线实体和影线
    body = abs(df['Close'] - df['Open'])
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    total_range = df['High'] - df['Low']
    
    # 锤子线 (Hammer)
    is_bullish = df['Close'] > df['Open']
    hammer_condition = (
        (lower_shadow > 2 * body) & 
        (upper_shadow < body) & 
        is_bullish
    )
    result['Hammer'] = hammer_condition.astype(int)
    
    # 上吊线 (Hanging Man)
    hanging_man_condition = (
        (lower_shadow > 2 * body) & 
        (upper_shadow < body) & 
        (~is_bullish)
    )
    result['Hanging_Man'] = hanging_man_condition.astype(int)
    
    # 十字星 (Doji)
    doji_condition = body < (0.1 * total_range)
    result['Doji'] = doji_condition.astype(int)
    
    # 看涨吞没 (Bullish Engulfing)
    bullish_engulfing = (
        (df['Close'] > df['Open']) &  # 当前为阳线
        (df['Close'].shift(1) < df['Open'].shift(1)) &  # 前一根为阴线
        (df['Open'] < df['Close'].shift(1)) &  # 当前开盘价低于前一根收盘价
        (df['Close'] > df['Open'].shift(1))  # 当前收盘价高于前一根开盘价
    )
    result['Bullish_Engulfing'] = bullish_engulfing.astype(int)
    
    # 看跌吞没 (Bearish Engulfing)
    bearish_engulfing = (
        (df['Close'] < df['Open']) &  # 当前为阴线
        (df['Close'].shift(1) > df['Open'].shift(1)) &  # 前一根为阳线
        (df['Open'] > df['Close'].shift(1)) &  # 当前开盘价高于前一根收盘价
        (df['Close'] < df['Open'].shift(1))  # 当前收盘价低于前一根开盘价
    )
    result['Bearish_Engulfing'] = bearish_engulfing.astype(int)
    
    return result

def calculate_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算季节性特征
    
    Args:
        df: 包含日期索引的DataFrame
    
    Returns:
        包含季节性特征的DataFrame
    """
    result = df.copy()
    
    # 确保索引是日期类型
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame索引必须是DatetimeIndex类型")
    
    # 时间特征
    result['DayOfWeek'] = df.index.dayofweek  # 0-6 (周一=0)
    result['Month'] = df.index.month  # 1-12
    result['Quarter'] = df.index.quarter  # 1-4
    result['DayOfYear'] = df.index.dayofyear  # 1-366
    result['WeekOfYear'] = df.index.isocalendar().week  # 1-53
    
    # 是否为月初/月末
    result['IsMonthStart'] = df.index.is_month_start.astype(int)
    result['IsMonthEnd'] = df.index.is_month_end.astype(int)
    
    # 是否为季初/季末
    result['IsQuarterStart'] = df.index.is_quarter_start.astype(int)
    result['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
    
    # 是否为年初/年末
    result['IsYearStart'] = df.index.is_year_start.astype(int)
    result['IsYearEnd'] = df.index.is_year_end.astype(int)
    
    return result

def calculate_technical_features(df: pd.DataFrame, 
                               rsi_period: int = 14,
                               macd_fast: int = 12,
                               macd_slow: int = 26,
                               macd_signal: int = 9,
                               bb_period: int = 20,
                               bb_std: float = 2,
                               stoch_period: int = 14,
                               atr_period: int = 14,
                               include_patterns: bool = True,
                               include_seasonal: bool = True) -> pd.DataFrame:
    """
    计算综合技术指标特征
    
    Args:
        df: 包含OHLC数据的DataFrame
        rsi_period: RSI周期
        macd_fast: MACD快线周期
        macd_slow: MACD慢线周期
        macd_signal: MACD信号线周期
        bb_period: 布林带周期
        bb_std: 布林带标准差倍数
        stoch_period: 随机指标周期
        atr_period: ATR周期
        include_patterns: 是否包含K线形态
        include_seasonal: 是否包含季节性特征
    
    Returns:
        包含所有技术指标的DataFrame
    """
    result = df.copy()
    
    # 确保必要的列存在
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 计算RSI
    result['RSI'] = calculate_rsi(df['Close'], period=rsi_period)
    
    # 计算MACD
    macd_df = calculate_macd(df['Close'], fast_period=macd_fast, 
                           slow_period=macd_slow, signal_period=macd_signal)
    result = pd.concat([result, macd_df], axis=1)
    
    # 计算布林带
    bb_df = calculate_bollinger_bands(df['Close'], period=bb_period, std_dev=bb_std)
    result = pd.concat([result, bb_df], axis=1)
    
    # 计算随机指标
    stoch_df = calculate_stochastic(df['High'], df['Low'], df['Close'], 
                                  k_period=stoch_period, d_period=3)
    result = pd.concat([result, stoch_df], axis=1)
    
    # 计算威廉指标
    result['Williams_R'] = calculate_williams_r(df['High'], df['Low'], df['Close'], 
                                              period=stoch_period)
    
    # 计算ATR
    result['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], period=atr_period)
    
    # K线形态检测
    if include_patterns:
        result = detect_candlestick_patterns(result)
    
    # 季节性特征
    if include_seasonal:
        result = calculate_seasonal_features(result)
    
    return result

def get_technical_feature_summary(df: pd.DataFrame) -> dict:
    """
    获取技术指标特征摘要
    
    Args:
        df: 包含技术指标的DataFrame
    
    Returns:
        特征摘要字典
    """
    summary = {}
    
    # RSI分析
    if 'RSI' in df.columns:
        rsi = df['RSI'].dropna()
        summary['RSI'] = {
            '当前值': rsi.iloc[-1] if len(rsi) > 0 else None,
            '平均值': rsi.mean(),
            '超买次数': (rsi > 70).sum(),
            '超卖次数': (rsi < 30).sum()
        }
    
    # MACD分析
    if 'MACD' in df.columns:
        macd = df['MACD'].dropna()
        macd_signal = df['MACD_Signal'].dropna()
        summary['MACD'] = {
            '当前MACD': macd.iloc[-1] if len(macd) > 0 else None,
            '当前信号线': macd_signal.iloc[-1] if len(macd_signal) > 0 else None,
            '金叉次数': ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).sum(),
            '死叉次数': ((macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))).sum()
        }
    
    # 布林带分析
    if 'BB_Position' in df.columns:
        bb_pos = df['BB_Position'].dropna()
        summary['布林带'] = {
            '当前位置': bb_pos.iloc[-1] if len(bb_pos) > 0 else None,
            '突破上轨次数': (bb_pos > 1).sum(),
            '突破下轨次数': (bb_pos < 0).sum()
        }
    
    # K线形态统计
    pattern_cols = ['Hammer', 'Hanging_Man', 'Doji', 'Bullish_Engulfing', 'Bearish_Engulfing']
    pattern_stats = {}
    for col in pattern_cols:
        if col in df.columns:
            pattern_stats[col] = df[col].sum()
    
    if pattern_stats:
        summary['K线形态'] = pattern_stats
    
    return summary 