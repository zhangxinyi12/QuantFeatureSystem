#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成交量衍生特征计算模块
提供各种成交量相关的技术指标和特征
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VolumeFeatures:
    """成交量特征计算器"""
    
    def __init__(self):
        """初始化成交量特征计算器"""
        self.feature_names = []
        
    def calculate_volume_ma(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        计算成交量移动平均
        
        Args:
            df: 包含Volume列的DataFrame
            windows: 移动平均窗口列表
            
        Returns:
            添加了成交量移动平均特征的DataFrame
        """
        result = df.copy()
        
        for window in windows:
            col_name = f'Volume_MA_{window}'
            result[col_name] = df['Volume'].rolling(window=window).mean()
            self.feature_names.append(col_name)
            
        return result
    
    def calculate_volume_ratio(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        计算成交量比率
        
        Args:
            df: 包含Volume列的DataFrame
            windows: 计算窗口列表
            
        Returns:
            添加了成交量比率特征的DataFrame
        """
        result = df.copy()
        
        for window in windows:
            # 当前成交量与移动平均的比率
            volume_ma = df['Volume'].rolling(window=window).mean()
            col_name = f'Volume_Ratio_{window}'
            result[col_name] = df['Volume'] / volume_ma
            self.feature_names.append(col_name)
            
        return result
    
    def calculate_volume_momentum(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        计算成交量动量
        
        Args:
            df: 包含Volume列的DataFrame
            windows: 计算窗口列表
            
        Returns:
            添加了成交量动量特征的DataFrame
        """
        result = df.copy()
        
        for window in windows:
            # 成交量变化率
            col_name = f'Volume_Momentum_{window}'
            result[col_name] = df['Volume'].pct_change(window)
            self.feature_names.append(col_name)
            
        return result
    
    def calculate_vwap(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        计算成交量加权平均价格 (VWAP)
        
        Args:
            df: 包含OHLCV数据的DataFrame
            window: 计算窗口
            
        Returns:
            添加了VWAP特征的DataFrame
        """
        result = df.copy()
        
        # 计算典型价格
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        
        # 计算VWAP
        vwap = (typical_price * df['Volume']).rolling(window=window).sum() / \
               df['Volume'].rolling(window=window).sum()
        
        result[f'VWAP_{window}'] = vwap
        self.feature_names.append(f'VWAP_{window}')
        
        return result
    
    def calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算累积/派发线 (A/D Line)
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了A/D Line特征的DataFrame
        """
        result = df.copy()
        
        # 计算资金流量乘数
        high_low = df['High'] - df['Low']
        close_low = df['Close'] - df['Low']
        high_close = df['High'] - df['Close']
        
        # 避免除零错误
        high_low = high_low.replace(0, 1)
        
        money_flow_multiplier = ((close_low - high_close) / high_low)
        money_flow_volume = money_flow_multiplier * df['Volume']
        
        # 计算累积/派发线
        result['AD_Line'] = money_flow_volume.cumsum()
        self.feature_names.append('AD_Line')
        
        return result
    
    def calculate_volume_price_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算量价趋势 (VPT)
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了VPT特征的DataFrame
        """
        result = df.copy()
        
        # 计算价格变化百分比
        price_change_pct = df['Close'].pct_change()
        
        # 计算VPT
        vpt = (price_change_pct * df['Volume']).cumsum()
        result['VPT'] = vpt
        self.feature_names.append('VPT')
        
        return result
    
    def calculate_volume_features(self, df: pd.DataFrame, 
                                ma_windows: List[int] = [5, 10, 20],
                                ratio_windows: List[int] = [5, 10, 20],
                                momentum_windows: List[int] = [5, 10, 20],
                                vwap_window: int = 20) -> pd.DataFrame:
        """
        计算所有成交量特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            ma_windows: 移动平均窗口列表
            ratio_windows: 比率计算窗口列表
            momentum_windows: 动量计算窗口列表
            vwap_window: VWAP计算窗口
            
        Returns:
            包含所有成交量特征的DataFrame
        """
        logger.info("开始计算成交量特征...")
        
        result = df.copy()
        
        # 计算各类成交量特征
        result = self.calculate_volume_ma(result, ma_windows)
        result = self.calculate_volume_ratio(result, ratio_windows)
        result = self.calculate_volume_momentum(result, momentum_windows)
        result = self.calculate_vwap(result, vwap_window)
        result = self.calculate_accumulation_distribution(result)
        result = self.calculate_volume_price_trend(result)
        
        logger.info(f"成交量特征计算完成，共生成 {len(self.feature_names)} 个特征")
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """获取所有生成的特征名称"""
        return self.feature_names.copy() 