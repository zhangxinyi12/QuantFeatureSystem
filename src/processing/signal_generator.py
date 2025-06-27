#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易信号生成模块
提供多种信号生成策略、风险控制和组合管理功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """信号类型枚举"""
    LONG = 1      # 做多信号
    SHORT = -1    # 做空信号
    NEUTRAL = 0   # 中性信号

class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = 0.1     # 低风险
    MEDIUM = 0.3  # 中等风险
    HIGH = 0.5    # 高风险

class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.6,
                 risk_level: RiskLevel = RiskLevel.MEDIUM,
                 max_position_size: float = 1.0):
        """
        初始化信号生成器
        
        Args:
            confidence_threshold: 信号置信度阈值
            risk_level: 风险等级
            max_position_size: 最大仓位大小
        """
        self.confidence_threshold = confidence_threshold
        self.risk_level = risk_level
        self.max_position_size = max_position_size
        self.signal_history = []
        
    def generate_signals_from_predictions(self, 
                                        df: pd.DataFrame,
                                        predictions: np.ndarray,
                                        model_type: str = 'regression',
                                        signal_method: str = 'threshold') -> pd.DataFrame:
        """
        从模型预测生成交易信号
        
        Args:
            df: 数据DataFrame
            predictions: 模型预测结果
            model_type: 模型类型 ('regression' or 'classification')
            signal_method: 信号生成方法
            
        Returns:
            包含交易信号的DataFrame
        """
        result = df.copy()
        
        if model_type == 'regression':
            signals = self._generate_regression_signals(predictions, signal_method)
        else:
            signals = self._generate_classification_signals(predictions, signal_method)
        
        # 添加信号列
        result['Signal'] = signals['signal']
        result['Confidence'] = signals['confidence']
        result['Position_Size'] = signals['position_size']
        result['Signal_Type'] = signals['signal_type']
        
        # 计算信号强度
        result['Signal_Strength'] = self._calculate_signal_strength(result)
        
        return result
    
    def _generate_regression_signals(self, 
                                   predictions: np.ndarray,
                                   method: str) -> Dict[str, np.ndarray]:
        """生成回归模型信号"""
        if method == 'threshold':
            return self._threshold_based_signals(predictions)
        elif method == 'percentile':
            return self._percentile_based_signals(predictions)
        elif method == 'zscore':
            return self._zscore_based_signals(predictions)
        else:
            logger.warning(f"未知的信号生成方法: {method}，使用阈值方法")
            return self._threshold_based_signals(predictions)
    
    def _generate_classification_signals(self, 
                                       predictions: np.ndarray,
                                       method: str) -> Dict[str, np.ndarray]:
        """生成分类模型信号"""
        if method == 'probability':
            return self._probability_based_signals(predictions)
        elif method == 'ensemble':
            return self._ensemble_based_signals(predictions)
        else:
            logger.warning(f"未知的信号生成方法: {method}，使用概率方法")
            return self._probability_based_signals(predictions)
    
    def _threshold_based_signals(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """基于阈值的信号生成"""
        # 计算信号
        signals = np.where(predictions > self.confidence_threshold, 1,
                          np.where(predictions < -self.confidence_threshold, -1, 0))
        
        # 计算置信度
        confidence = np.abs(predictions)
        
        # 计算仓位大小
        position_size = np.minimum(np.abs(predictions) * self.risk_level.value, 
                                 self.max_position_size)
        
        # 信号类型
        signal_type = np.where(signals == 1, SignalType.LONG.value,
                              np.where(signals == -1, SignalType.SHORT.value, 
                                     SignalType.NEUTRAL.value))
        
        return {
            'signal': signals,
            'confidence': confidence,
            'position_size': position_size,
            'signal_type': signal_type
        }
    
    def _percentile_based_signals(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """基于百分位的信号生成"""
        # 计算滚动百分位
        window = 252  # 一年
        percentiles = pd.Series(predictions).rolling(window).rank(pct=True)
        
        # 生成信号
        signals = np.where(percentiles > 0.8, 1,
                          np.where(percentiles < 0.2, -1, 0))
        
        # 置信度基于百分位距离
        confidence = np.where(percentiles > 0.5, 
                             percentiles, 
                             1 - percentiles)
        
        # 仓位大小
        position_size = confidence * self.risk_level.value
        
        # 信号类型
        signal_type = np.where(signals == 1, SignalType.LONG.value,
                              np.where(signals == -1, SignalType.SHORT.value, 
                                     SignalType.NEUTRAL.value))
        
        return {
            'signal': signals,
            'confidence': confidence,
            'position_size': position_size,
            'signal_type': signal_type
        }
    
    def _zscore_based_signals(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """基于Z分数的信号生成"""
        # 计算滚动Z分数
        window = 252
        rolling_mean = pd.Series(predictions).rolling(window).mean()
        rolling_std = pd.Series(predictions).rolling(window).std()
        zscores = (predictions - rolling_mean) / rolling_std
        
        # 生成信号
        signals = np.where(zscores > 1.5, 1,
                          np.where(zscores < -1.5, -1, 0))
        
        # 置信度
        confidence = np.minimum(np.abs(zscores) / 3, 1)
        
        # 仓位大小
        position_size = confidence * self.risk_level.value
        
        # 信号类型
        signal_type = np.where(signals == 1, SignalType.LONG.value,
                              np.where(signals == -1, SignalType.SHORT.value, 
                                     SignalType.NEUTRAL.value))
        
        return {
            'signal': signals,
            'confidence': confidence,
            'position_size': position_size,
            'signal_type': signal_type
        }
    
    def _probability_based_signals(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """基于概率的信号生成"""
        # 假设predictions是概率值 [0, 1]
        signals = np.where(predictions > 0.5 + self.confidence_threshold/2, 1,
                          np.where(predictions < 0.5 - self.confidence_threshold/2, -1, 0))
        
        # 置信度
        confidence = np.where(predictions > 0.5,
                             predictions,
                             1 - predictions)
        
        # 仓位大小
        position_size = confidence * self.risk_level.value
        
        # 信号类型
        signal_type = np.where(signals == 1, SignalType.LONG.value,
                              np.where(signals == -1, SignalType.SHORT.value, 
                                     SignalType.NEUTRAL.value))
        
        return {
            'signal': signals,
            'confidence': confidence,
            'position_size': position_size,
            'signal_type': signal_type
        }
    
    def _ensemble_based_signals(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """基于集成模型的信号生成"""
        # 假设predictions是多个模型的预测结果
        if len(predictions.shape) > 1:
            # 多个模型的情况
            ensemble_pred = np.mean(predictions, axis=1)
            ensemble_std = np.std(predictions, axis=1)
            
            # 信号基于集成预测
            signals = np.where(ensemble_pred > self.confidence_threshold, 1,
                              np.where(ensemble_pred < -self.confidence_threshold, -1, 0))
            
            # 置信度基于预测一致性
            confidence = 1 / (1 + ensemble_std)
            
        else:
            # 单个模型的情况
            signals = np.where(predictions > self.confidence_threshold, 1,
                              np.where(predictions < -self.confidence_threshold, -1, 0))
            confidence = np.abs(predictions)
        
        # 仓位大小
        position_size = confidence * self.risk_level.value
        
        # 信号类型
        signal_type = np.where(signals == 1, SignalType.LONG.value,
                              np.where(signals == -1, SignalType.SHORT.value, 
                                     SignalType.NEUTRAL.value))
        
        return {
            'signal': signals,
            'confidence': confidence,
            'position_size': position_size,
            'signal_type': signal_type
        }
    
    def _calculate_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算信号强度"""
        # 基于置信度和仓位大小的综合强度
        strength = df['Confidence'] * df['Position_Size']
        
        # 考虑信号持续性
        if 'Signal' in df.columns:
            signal_persistence = df['Signal'].rolling(5).sum().abs() / 5
            strength = strength * (1 + signal_persistence * 0.2)
        
        return strength
    
    def apply_risk_controls(self, df: pd.DataFrame,
                           max_drawdown: float = 0.2,
                           volatility_target: float = 0.15) -> pd.DataFrame:
        """
        应用风险控制
        
        Args:
            df: 包含信号的DataFrame
            max_drawdown: 最大回撤限制
            volatility_target: 目标波动率
            
        Returns:
            应用风险控制后的DataFrame
        """
        result = df.copy()
        
        # 计算累积收益
        if 'ClosePrice' in df.columns:
            returns = df['ClosePrice'].pct_change()
        elif 'Close' in df.columns:
            returns = df['Close'].pct_change()
        else:
            logger.warning("无法找到价格数据，跳过风险控制")
            return result
        
        # 计算策略收益
        strategy_returns = returns * df['Signal'].shift(1)
        
        # 计算累积收益和回撤
        cumulative_returns = (1 + strategy_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        # 回撤控制
        drawdown_control = np.where(drawdown < -max_drawdown, 0, 1)
        
        # 波动率控制
        rolling_vol = strategy_returns.rolling(20).std() * np.sqrt(252)
        vol_control = np.where(rolling_vol > volatility_target, 
                              volatility_target / rolling_vol, 1)
        
        # 应用风险控制
        result['Risk_Adjusted_Signal'] = (df['Signal'] * 
                                         drawdown_control * 
                                         vol_control)
        
        result['Risk_Adjusted_Position'] = (df['Position_Size'] * 
                                           drawdown_control * 
                                           vol_control)
        
        return result
    
    def generate_portfolio_signals(self, 
                                 signals_dict: Dict[str, pd.DataFrame],
                                 weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        生成组合信号
        
        Args:
            signals_dict: 各资产信号字典
            weights: 权重字典
            
        Returns:
            组合信号DataFrame
        """
        if not signals_dict:
            return pd.DataFrame()
        
        # 默认等权重
        if weights is None:
            n_assets = len(signals_dict)
            weights = {asset: 1.0/n_assets for asset in signals_dict.keys()}
        
        # 合并所有信号
        portfolio_signals = []
        for asset, signals_df in signals_dict.items():
            if 'Signal' in signals_df.columns:
                weighted_signal = signals_df['Signal'] * weights.get(asset, 0)
                portfolio_signals.append(weighted_signal)
        
        if not portfolio_signals:
            return pd.DataFrame()
        
        # 计算组合信号
        combined_signals = pd.concat(portfolio_signals, axis=1).sum(axis=1)
        
        # 创建结果DataFrame
        result = pd.DataFrame({
            'Portfolio_Signal': combined_signals,
            'Portfolio_Confidence': np.abs(combined_signals),
            'Portfolio_Position': np.minimum(np.abs(combined_signals), self.max_position_size)
        })
        
        return result
    
    def get_signal_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取信号统计信息"""
        if 'Signal' not in df.columns:
            return {}
        
        signals = df['Signal']
        
        stats = {
            'total_signals': len(signals[signals != 0]),
            'long_signals': len(signals[signals == 1]),
            'short_signals': len(signals[signals == -1]),
            'signal_frequency': len(signals[signals != 0]) / len(signals),
            'avg_confidence': df.get('Confidence', pd.Series(0)).mean(),
            'avg_position_size': df.get('Position_Size', pd.Series(0)).mean()
        }
        
        return stats 