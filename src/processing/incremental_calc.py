#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量计算模块
提供增量式的特征计算功能，适用于实时数据处理和增量更新
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from collections import deque
import pickle
import os

logger = logging.getLogger('incremental_calculator')

class IncrementalCalculator:
    """增量计算器"""
    
    def __init__(self, state_file: Optional[str] = None):
        """
        初始化增量计算器
        
        Args:
            state_file: 状态保存文件路径
        """
        self.state = {}
        self.state_file = state_file
        self.logger = logging.getLogger(__name__)
        
        # 加载已有状态
        if state_file and os.path.exists(state_file):
            self.load_state()
    
    def save_state(self) -> None:
        """保存计算状态"""
        if self.state_file:
            try:
                with open(self.state_file, 'wb') as f:
                    pickle.dump(self.state, f)
                self.logger.info(f"状态已保存到: {self.state_file}")
            except Exception as e:
                self.logger.error(f"保存状态失败: {e}")
    
    def load_state(self) -> None:
        """加载计算状态"""
        if self.state_file and os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    self.state = pickle.load(f)
                self.logger.info(f"状态已从 {self.state_file} 加载")
            except Exception as e:
                self.logger.error(f"加载状态失败: {e}")
                self.state = {}
    
    def update_moving_average(
        self, 
        new_data: pd.Series, 
        window: int,
        feature_name: Optional[str] = None
    ) -> float:
        """
        增量更新移动平均
        
        Args:
            new_data: 新数据
            window: 窗口大小
            feature_name: 特征名称
            
        Returns:
            float: 更新后的移动平均值
        """
        if feature_name is None:
            feature_name = f'MA_{window}'
        
        if feature_name not in self.state:
            # 初次计算
            if len(new_data) >= window:
                ma = new_data.rolling(window=window).mean().iloc[-1]
                self.state[feature_name] = ma
            else:
                # 数据不足，使用简单平均
                ma = new_data.mean()
                self.state[feature_name] = ma
        else:
            # 增量更新
            prev_ma = self.state[feature_name]
            new_value = new_data.iloc[-1]
            
            # 使用递推公式更新
            ma = prev_ma + (new_value - prev_ma) / window
            self.state[feature_name] = ma
        
        return self.state[feature_name]
    
    def update_exponential_moving_average(
        self, 
        new_data: pd.Series, 
        span: int,
        feature_name: Optional[str] = None
    ) -> float:
        """
        增量更新指数移动平均
        
        Args:
            new_data: 新数据
            span: 平滑系数
            feature_name: 特征名称
            
        Returns:
            float: 更新后的EMA值
        """
        if feature_name is None:
            feature_name = f'EMA_{span}'
        
        alpha = 2 / (span + 1)
        
        if feature_name not in self.state:
            # 初次计算
            ema = new_data.iloc[-1]
            self.state[feature_name] = ema
        else:
            # 增量更新
            prev_ema = self.state[feature_name]
            new_value = new_data.iloc[-1]
            
            ema = alpha * new_value + (1 - alpha) * prev_ema
            self.state[feature_name] = ema
        
        return self.state[feature_name]
    
    def update_volatility(
        self, 
        new_data: pd.Series, 
        window: int,
        feature_name: Optional[str] = None
    ) -> float:
        """
        增量更新波动率
        
        Args:
            new_data: 新数据
            window: 窗口大小
            feature_name: 特征名称
            
        Returns:
            float: 更新后的波动率
        """
        if feature_name is None:
            feature_name = f'Volatility_{window}'
        
        if feature_name not in self.state:
            # 初次计算
            if len(new_data) >= window:
                vol = new_data.rolling(window=window).std().iloc[-1]
                self.state[feature_name] = vol
            else:
                vol = new_data.std()
                self.state[feature_name] = vol
        else:
            # 增量更新（简化版本）
            prev_vol = self.state[feature_name]
            new_value = new_data.iloc[-1]
            
            # 使用EWMA更新波动率
            alpha = 2 / (window + 1)
            vol = np.sqrt(alpha * new_value**2 + (1 - alpha) * prev_vol**2)
            self.state[feature_name] = vol
        
        return self.state[feature_name]
    
    def update_max_min(
        self, 
        new_data: pd.Series,
        feature_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        增量更新最大值和最小值
        
        Args:
            new_data: 新数据
            feature_name: 特征名称前缀
            
        Returns:
            Dict: 包含最大值和最小值的字典
        """
        if feature_name is None:
            feature_name = 'Extremes'
        
        max_key = f'{feature_name}_max'
        min_key = f'{feature_name}_min'
        
        new_value = new_data.iloc[-1]
        
        # 更新最大值
        if max_key not in self.state:
            self.state[max_key] = new_value
        else:
            self.state[max_key] = max(self.state[max_key], new_value)
        
        # 更新最小值
        if min_key not in self.state:
            self.state[min_key] = new_value
        else:
            self.state[min_key] = min(self.state[min_key], new_value)
        
        return {
            'max': self.state[max_key],
            'min': self.state[min_key]
        }
    
    def update_cumulative_sum(
        self, 
        new_data: pd.Series,
        feature_name: Optional[str] = None
    ) -> float:
        """
        增量更新累积和
        
        Args:
            new_data: 新数据
            feature_name: 特征名称
            
        Returns:
            float: 更新后的累积和
        """
        if feature_name is None:
            feature_name = 'CumSum'
        
        new_value = new_data.iloc[-1]
        
        if feature_name not in self.state:
            self.state[feature_name] = new_value
        else:
            self.state[feature_name] += new_value
        
        return self.state[feature_name]
    
    def update_rolling_window(
        self, 
        new_data: pd.Series, 
        window: int,
        feature_name: Optional[str] = None
    ) -> deque:
        """
        增量更新滚动窗口
        
        Args:
            new_data: 新数据
            window: 窗口大小
            feature_name: 特征名称
            
        Returns:
            deque: 更新后的滚动窗口
        """
        if feature_name is None:
            feature_name = f'RollingWindow_{window}'
        
        new_value = new_data.iloc[-1]
        
        if feature_name not in self.state:
            # 初始化滚动窗口
            self.state[feature_name] = deque(maxlen=window)
        
        # 添加新值
        self.state[feature_name].append(new_value)
        
        return self.state[feature_name]
    
    def get_rolling_statistics(
        self, 
        window_name: str
    ) -> Dict[str, float]:
        """
        获取滚动窗口的统计信息
        
        Args:
            window_name: 窗口名称
            
        Returns:
            Dict: 统计信息字典
        """
        if window_name not in self.state:
            return {}
        
        window_data = list(self.state[window_name])
        
        if not window_data:
            return {}
        
        return {
            'mean': np.mean(window_data),
            'std': np.std(window_data),
            'min': np.min(window_data),
            'max': np.max(window_data),
            'count': len(window_data)
        }
    
    def update_correlation(
        self, 
        series1: pd.Series, 
        series2: pd.Series, 
        window: int,
        feature_name: Optional[str] = None
    ) -> float:
        """
        增量更新相关系数（简化版本）
        
        Args:
            series1: 第一个序列
            series2: 第二个序列
            window: 窗口大小
            feature_name: 特征名称
            
        Returns:
            float: 更新后的相关系数
        """
        if feature_name is None:
            feature_name = f'Correlation_{window}'
        
        # 获取最新的值
        val1 = series1.iloc[-1]
        val2 = series2.iloc[-1]
        
        # 使用滚动窗口存储历史数据
        window1_name = f'{feature_name}_series1'
        window2_name = f'{feature_name}_series2'
        
        self.update_rolling_window(series1, window, window1_name)
        self.update_rolling_window(series2, window, window2_name)
        
        # 计算相关系数
        if len(self.state[window1_name]) >= 2:
            corr = np.corrcoef(list(self.state[window1_name]), 
                             list(self.state[window2_name]))[0, 1]
            self.state[feature_name] = corr
            return corr
        else:
            return 0.0
    
    def reset_state(self, feature_name: Optional[str] = None) -> None:
        """
        重置状态
        
        Args:
            feature_name: 要重置的特征名称，如果为None则重置所有状态
        """
        if feature_name is None:
            self.state = {}
            self.logger.info("所有状态已重置")
        else:
            if feature_name in self.state:
                del self.state[feature_name]
                self.logger.info(f"特征 {feature_name} 的状态已重置")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        获取状态摘要
        
        Returns:
            Dict: 状态摘要
        """
        summary = {
            'total_features': len(self.state),
            'feature_names': list(self.state.keys()),
            'state_size': len(pickle.dumps(self.state))
        }
        
        # 添加每个特征的详细信息
        for name, value in self.state.items():
            if isinstance(value, deque):
                summary[name] = {
                    'type': 'rolling_window',
                    'length': len(value),
                    'maxlen': value.maxlen
                }
            elif isinstance(value, (int, float)):
                summary[name] = {
                    'type': 'numeric',
                    'value': value
                }
            else:
                summary[name] = {
                    'type': 'other',
                    'value_type': type(value).__name__
                }
        
        return summary
