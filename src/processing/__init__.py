#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理核心包
提供时序处理、增量计算、特征工程、特征筛选、模型训练等功能
"""

# 导入时序处理模块
from .timeseries import TimeSeriesProcessor

# 导入增量计算模块
from .incremental_calc import IncrementalCalculator

# 导入特征工程模块
from .feature_engine import (
    QuantFeatureEngine, 
    TechnicalIndicators, 
    OrderBookFeatures,
    VolumeFeatures,
    FundamentalFeatures
)

# 导入特征筛选模块
from .feature_selection import FeatureSelector

# 导入模型训练模块
from .model_training import ModelTrainer

# 版本信息
__version__ = "1.0.0"
__author__ = 'QuantFeatureSystem Team'

# 主要类和函数
__all__ = [
    # 时序处理
    'TimeSeriesProcessor',
    
    # 增量计算
    'IncrementalCalculator',
    
    # 特征工程
    'QuantFeatureEngine',
    'TechnicalIndicators',
    'OrderBookFeatures', 
    'VolumeFeatures',
    'FundamentalFeatures',
    
    # 特征筛选
    'FeatureSelector',
    
    # 模型训练
    'ModelTrainer'
]
