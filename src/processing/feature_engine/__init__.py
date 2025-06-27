#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征引擎包
提供价格特征、技术指标、订单簿特征、成交量特征、基本面特征等计算功能
"""

from .price_features import QuantFeatureEngine
from .technical_indicators import TechnicalIndicators
from .order_book_features import OrderBookFeatures
from .volume_features import VolumeFeatures
from .fundamental_features import FundamentalFeatures

__all__ = [
    'QuantFeatureEngine',
    'TechnicalIndicators', 
    'OrderBookFeatures',
    'VolumeFeatures',
    'FundamentalFeatures'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'QuantFeatureSystem Team'
