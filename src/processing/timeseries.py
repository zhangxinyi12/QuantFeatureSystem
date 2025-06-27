#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序数据处理模块
专注于时序数据的通用处理功能，如重采样、对齐、滑动窗口等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger('timeseries_processor')

class TimeSeriesProcessor:
    """时序数据处理器"""
    
    def __init__(self):
        """初始化处理器"""
        self.logger = logging.getLogger(__name__)
    
    def resample_data(
        self, 
        df: pd.DataFrame,
        freq: str = 'D',
        agg_method: str = 'ohlc',
        date_col: str = 'TradingDay',
        price_cols: Optional[List[str]] = None,
        volume_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        数据重采样
        
        Args:
            df: 输入数据DataFrame
            freq: 重采样频率 ('D', 'W', 'M', 'Q', 'Y')
            agg_method: 聚合方法 ('ohlc', 'last', 'mean', 'sum')
            date_col: 日期列名
            price_cols: 价格列名列表
            volume_cols: 成交量列名列表
            
        Returns:
            pd.DataFrame: 重采样后的数据
        """
        if date_col not in df.columns:
            raise ValueError(f"日期列 {date_col} 不存在")
        
        # 设置日期索引
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy.set_index(date_col, inplace=True)
        
        # 按股票代码分组重采样
        result_dfs = []
        
        for secu_code in df_copy['SecuCode'].unique():
            stock_data = df_copy[df_copy['SecuCode'] == secu_code].copy()
            
            if agg_method == 'ohlc':
                # OHLC聚合
                resampled = stock_data.resample(freq).agg({
                    'OpenPrice': 'first',
                    'HighPrice': 'max',
                    'LowPrice': 'min',
                    'ClosePrice': 'last',
                    'TurnoverVolume': 'sum',
                    'TurnoverValue': 'sum'
                })
            elif agg_method == 'last':
                # 取最后一个值
                resampled = stock_data.resample(freq).last()
            elif agg_method == 'mean':
                # 取平均值
                resampled = stock_data.resample(freq).mean()
            elif agg_method == 'sum':
                # 求和
                resampled = stock_data.resample(freq).sum()
            else:
                raise ValueError(f"不支持的聚合方法: {agg_method}")
            
            resampled['SecuCode'] = secu_code
            result_dfs.append(resampled)
        
        result = pd.concat(result_dfs, ignore_index=False)
        result.reset_index(inplace=True)
        
        self.logger.info(f"数据重采样完成，频率: {freq}, 聚合方法: {agg_method}")
        return result
    
    def align_data(
        self, 
        df: pd.DataFrame,
        date_col: str = 'TradingDay',
        secu_col: str = 'SecuCode',
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        数据对齐（确保所有股票在同一时间点都有数据）
        
        Args:
            df: 输入数据DataFrame
            date_col: 日期列名
            secu_col: 股票代码列名
            fill_method: 填充方法 ('ffill', 'bfill', 'interpolate')
            
        Returns:
            pd.DataFrame: 对齐后的数据
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # 创建完整的时间索引
        all_dates = df_copy[date_col].unique()
        all_secus = df_copy[secu_col].unique()
        
        # 创建完整的时间-股票组合
        date_secu_pairs = pd.MultiIndex.from_product(
            [all_dates, all_secus], 
            names=[date_col, secu_col]
        )
        
        # 重新索引并填充缺失值
        df_copy.set_index([date_col, secu_col], inplace=True)
        df_aligned = df_copy.reindex(date_secu_pairs)
        
        if fill_method == 'ffill':
            df_aligned = df_aligned.groupby(level=1).ffill()
        elif fill_method == 'bfill':
            df_aligned = df_aligned.groupby(level=1).bfill()
        elif fill_method == 'interpolate':
            df_aligned = df_aligned.groupby(level=1).interpolate()
        
        df_aligned.reset_index(inplace=True)
        
        self.logger.info(f"数据对齐完成，填充方法: {fill_method}")
        return df_aligned
    
    def rolling_window(
        self, 
        df: pd.DataFrame,
        window: int,
        min_periods: Optional[int] = None,
        center: bool = False,
        win_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        滑动窗口处理
        
        Args:
            df: 输入数据DataFrame
            window: 窗口大小
            min_periods: 最小观测数
            center: 是否居中
            win_type: 窗口类型
            
        Returns:
            pd.DataFrame: 滑动窗口处理后的数据
        """
        result_df = df.copy()
        
        # 数值列进行滑动窗口处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['SecuCode']:  # 排除非数值列
                result_df[f'{col}_rolling_mean'] = (
                    df[col].rolling(
                        window=window, 
                        min_periods=min_periods, 
                        center=center, 
                        win_type=win_type
                    ).mean()
                )
                result_df[f'{col}_rolling_std'] = (
                    df[col].rolling(
                        window=window, 
                        min_periods=min_periods, 
                        center=center, 
                        win_type=win_type
                    ).std()
                )
        
        self.logger.info(f"滑动窗口处理完成，窗口大小: {window}")
        return result_df
    
    def calculate_returns(
        self, 
        df: pd.DataFrame,
        price_col: str = 'ClosePrice',
        periods: List[int] = [1, 5, 20, 60],
        return_type: str = 'simple'
    ) -> pd.DataFrame:
        """
        计算收益率
        
        Args:
            df: 输入数据DataFrame
            price_col: 价格列名
            periods: 计算周期列表
            return_type: 收益率类型 ('simple', 'log')
            
        Returns:
            pd.DataFrame: 包含收益率的数据
        """
        result_df = df.copy()
        
        for secu_code in result_df['SecuCode'].unique():
            mask = result_df['SecuCode'] == secu_code
            stock_data = result_df[mask].copy().sort_values('TradingDay')
            
            if price_col in stock_data.columns:
                prices = stock_data[price_col]
                
                for period in periods:
                    if return_type == 'simple':
                        returns = prices.pct_change(periods=period)
                    elif return_type == 'log':
                        returns = np.log(prices / prices.shift(period))
                    else:
                        raise ValueError(f"不支持的收益率类型: {return_type}")
                    
                    result_df.loc[mask, f'Return_{period}d'] = returns
        
        self.logger.info(f"收益率计算完成，类型: {return_type}")
        return result_df
    
    def calculate_volatility(
        self, 
        df: pd.DataFrame,
        price_col: str = 'ClosePrice',
        windows: List[int] = [20, 60],
        method: str = 'std'
    ) -> pd.DataFrame:
        """
        计算波动率
        
        Args:
            df: 输入数据DataFrame
            price_col: 价格列名
            windows: 计算窗口列表
            method: 计算方法 ('std', 'parkinson', 'garman_klass')
            
        Returns:
            pd.DataFrame: 包含波动率的数据
        """
        result_df = df.copy()
        
        for secu_code in result_df['SecuCode'].unique():
            mask = result_df['SecuCode'] == secu_code
            stock_data = result_df[mask].copy().sort_values('TradingDay')
            
            if price_col in stock_data.columns:
                prices = stock_data[price_col]
                returns = prices.pct_change()
                
                for window in windows:
                    if method == 'std':
                        vol = returns.rolling(window=window).std() * np.sqrt(252)
                    elif method == 'parkinson':
                        # 需要High和Low价格
                        if 'HighPrice' in stock_data.columns and 'LowPrice' in stock_data.columns:
                            high = stock_data['HighPrice']
                            low = stock_data['LowPrice']
                            vol = np.sqrt(
                                (np.log(high / low) ** 2).rolling(window=window).mean() / 
                                (4 * np.log(2)) * 252
                            )
                        else:
                            vol = returns.rolling(window=window).std() * np.sqrt(252)
                    elif method == 'garman_klass':
                        # 需要OHLC价格
                        if all(col in stock_data.columns for col in ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice']):
                            open_price = stock_data['OpenPrice']
                            high = stock_data['HighPrice']
                            low = stock_data['LowPrice']
                            close = stock_data['ClosePrice']
                            
                            vol = np.sqrt(
                                (0.5 * np.log(high / low) ** 2 - 
                                 (2 * np.log(2) - 1) * np.log(close / open_price) ** 2
                                ).rolling(window=window).mean() * 252
                            )
                        else:
                            vol = returns.rolling(window=window).std() * np.sqrt(252)
                    else:
                        raise ValueError(f"不支持的波动率计算方法: {method}")
                    
                    result_df.loc[mask, f'Volatility_{window}d'] = vol
        
        self.logger.info(f"波动率计算完成，方法: {method}")
        return result_df
    
    def filter_trading_days(
        self, 
        df: pd.DataFrame,
        min_volume: Optional[float] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> pd.DataFrame:
        """
        过滤交易日
        
        Args:
            df: 输入数据DataFrame
            min_volume: 最小成交量
            min_price: 最小价格
            max_price: 最大价格
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        result_df = df.copy()
        
        # 成交量过滤
        if min_volume is not None and 'TurnoverVolume' in result_df.columns:
            result_df = result_df[result_df['TurnoverVolume'] >= min_volume]
        
        # 价格过滤
        if min_price is not None and 'ClosePrice' in result_df.columns:
            result_df = result_df[result_df['ClosePrice'] >= min_price]
        
        if max_price is not None and 'ClosePrice' in result_df.columns:
            result_df = result_df[result_df['ClosePrice'] <= max_price]
        
        self.logger.info(f"交易日过滤完成，剩余数据量: {len(result_df)}")
        return result_df
    
    def add_time_features(
        self, 
        df: pd.DataFrame,
        date_col: str = 'TradingDay'
    ) -> pd.DataFrame:
        """
        添加时间特征
        
        Args:
            df: 输入数据DataFrame
            date_col: 日期列名
            
        Returns:
            pd.DataFrame: 包含时间特征的数据
        """
        result_df = df.copy()
        result_df[date_col] = pd.to_datetime(result_df[date_col])
        
        # 基本时间特征
        result_df['Year'] = result_df[date_col].dt.year
        result_df['Month'] = result_df[date_col].dt.month
        result_df['Day'] = result_df[date_col].dt.day
        result_df['DayOfWeek'] = result_df[date_col].dt.dayofweek
        result_df['Quarter'] = result_df[date_col].dt.quarter
        result_df['DayOfYear'] = result_df[date_col].dt.dayofyear
        
        # 是否为月初/月末
        result_df['IsMonthStart'] = result_df[date_col].dt.is_month_start.astype(int)
        result_df['IsMonthEnd'] = result_df[date_col].dt.is_month_end.astype(int)
        
        # 是否为季初/季末
        result_df['IsQuarterStart'] = result_df[date_col].dt.is_quarter_start.astype(int)
        result_df['IsQuarterEnd'] = result_df[date_col].dt.is_quarter_end.astype(int)
        
        self.logger.info("时间特征添加完成")
        return result_df 