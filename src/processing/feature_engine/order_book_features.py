#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单簿特征模块
包含基于Level 2数据的订单簿特征计算
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict
import warnings

def calculate_order_flow_imbalance(df: pd.DataFrame, 
                                 bid_price_col: str = 'Bid_Price1',
                                 bid_size_col: str = 'Bid_Size1',
                                 ask_price_col: str = 'Ask_Price1',
                                 ask_size_col: str = 'Ask_Size1') -> pd.DataFrame:
    """
    计算订单流不平衡特征
    
    Args:
        df: 包含订单簿数据的DataFrame
        bid_price_col: 买一价列名
        bid_size_col: 买一量列名
        ask_price_col: 卖一价列名
        ask_size_col: 卖一量列名
    
    Returns:
        包含订单流不平衡特征的DataFrame
    """
    result = df.copy()
    
    # 检查必要的列是否存在
    required_cols = [bid_price_col, bid_size_col, ask_price_col, ask_size_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 计算中间价
    result['Mid_Price'] = (df[bid_price_col] + df[ask_price_col]) / 2
    
    # 计算买卖价差
    result['Spread'] = df[ask_price_col] - df[bid_price_col]
    result['Spread_Ratio'] = result['Spread'] / result['Mid_Price']
    
    # 订单流不平衡 (OFI)
    result['OFI'] = (df[bid_size_col] - df[ask_size_col]) / (df[bid_size_col] + df[ask_size_col])
    
    # 价格压力
    result['Price_Pressure'] = (df[ask_price_col] - result['Mid_Price']) / result['Spread']
    
    # 买卖压力比率
    result['Bid_Ask_Ratio'] = df[bid_size_col] / df[ask_size_col]
    
    # 订单簿深度
    result['Total_Depth'] = df[bid_size_col] + df[ask_size_col]
    
    return result

def calculate_order_book_imbalance(df: pd.DataFrame, 
                                 levels: int = 5,
                                 bid_price_prefix: str = 'Bid_Price',
                                 bid_size_prefix: str = 'Bid_Size',
                                 ask_price_prefix: str = 'Ask_Price',
                                 ask_size_prefix: str = 'Ask_Size') -> pd.DataFrame:
    """
    计算多档订单簿不平衡特征
    
    Args:
        df: 包含多档订单簿数据的DataFrame
        levels: 订单簿档数
        bid_price_prefix: 买价列名前缀
        bid_size_prefix: 买量列名前缀
        ask_price_prefix: 卖价列名前缀
        ask_size_prefix: 卖量列名前缀
    
    Returns:
        包含多档订单簿不平衡特征的DataFrame
    """
    result = df.copy()
    
    # 计算各档位的订单流不平衡
    for level in range(1, levels + 1):
        bid_price_col = f"{bid_price_prefix}{level}"
        bid_size_col = f"{bid_size_prefix}{level}"
        ask_price_col = f"{ask_price_prefix}{level}"
        ask_size_col = f"{ask_size_prefix}{level}"
        
        # 检查该档位的数据是否存在
        if all(col in df.columns for col in [bid_price_col, bid_size_col, ask_price_col, ask_size_col]):
            # 该档位的订单流不平衡
            result[f'OFI_Level_{level}'] = (
                (df[bid_size_col] - df[ask_size_col]) / 
                (df[bid_size_col] + df[ask_size_col])
            )
            
            # 该档位的价格压力
            mid_price = (df[bid_price_col] + df[ask_price_col]) / 2
            spread = df[ask_price_col] - df[bid_price_col]
            result[f'Price_Pressure_Level_{level}'] = (df[ask_price_col] - mid_price) / spread
    
    # 计算累积订单流不平衡
    ofi_cols = [col for col in result.columns if col.startswith('OFI_Level_')]
    if ofi_cols:
        result['Cumulative_OFI'] = result[ofi_cols].sum(axis=1)
    
    # 计算加权订单流不平衡 (按价格权重)
    weighted_ofi = 0
    total_weight = 0
    
    for level in range(1, levels + 1):
        bid_price_col = f"{bid_price_prefix}{level}"
        ask_price_col = f"{ask_price_prefix}{level}"
        ofi_col = f'OFI_Level_{level}'
        
        if all(col in result.columns for col in [bid_price_col, ask_price_col, ofi_col]):
            weight = 1 / level  # 越靠近最优价的档位权重越大
            weighted_ofi += result[ofi_col] * weight
            total_weight += weight
    
    if total_weight > 0:
        result['Weighted_OFI'] = weighted_ofi / total_weight
    
    return result

def calculate_microstructure_features(df: pd.DataFrame,
                                    price_col: str = 'Mid_Price',
                                    volume_col: str = 'Volume',
                                    window: int = 20) -> pd.DataFrame:
    """
    计算市场微观结构特征
    
    Args:
        df: 包含价格和成交量数据的DataFrame
        price_col: 价格列名
        volume_col: 成交量列名
        window: 计算窗口
    
    Returns:
        包含微观结构特征的DataFrame
    """
    result = df.copy()
    
    # 价格冲击
    result['Price_Impact'] = df[price_col].pct_change()
    
    # 成交量加权价格冲击
    result['Volume_Weighted_Price_Impact'] = (
        result['Price_Impact'] * df[volume_col]
    ).rolling(window=window).mean()
    
    # 价格波动率
    result['Price_Volatility'] = result['Price_Impact'].rolling(window=window).std()
    
    # 成交量波动率
    result['Volume_Volatility'] = df[volume_col].pct_change().rolling(window=window).std()
    
    # Kyle's Lambda (价格冲击系数)
    price_impact = result['Price_Impact'].abs()
    volume = df[volume_col]
    result['Kyle_Lambda'] = (
        (price_impact * volume).rolling(window=window).mean() / 
        (volume ** 2).rolling(window=window).mean()
    )
    
    # Amihud流动性指标
    result['Amihud_Illiquidity'] = (
        result['Price_Impact'].abs() / df[volume_col]
    ).rolling(window=window).mean()
    
    return result

def calculate_order_flow_metrics(df: pd.DataFrame,
                               bid_size_col: str = 'Bid_Size1',
                               ask_size_col: str = 'Ask_Size1',
                               volume_col: str = 'Volume',
                               window: int = 20) -> pd.DataFrame:
    """
    计算订单流指标
    
    Args:
        df: 包含订单流数据的DataFrame
        bid_size_col: 买一量列名
        ask_size_col: 卖一量列名
        volume_col: 成交量列名
        window: 计算窗口
    
    Returns:
        包含订单流指标的DataFrame
    """
    result = df.copy()
    
    # 买卖订单流比率
    result['Bid_Flow_Ratio'] = df[bid_size_col] / (df[bid_size_col] + df[ask_size_col])
    result['Ask_Flow_Ratio'] = df[ask_size_col] / (df[bid_size_col] + df[ask_size_col])
    
    # 订单流不平衡的移动平均
    result['OFI_MA'] = result['OFI'].rolling(window=window).mean()
    result['OFI_Std'] = result['OFI'].rolling(window=window).std()
    
    # 订单流Z分数
    result['OFI_ZScore'] = (result['OFI'] - result['OFI_MA']) / result['OFI_Std']
    
    # 大单检测 (基于订单流不平衡)
    result['Large_Order_Indicator'] = (result['OFI_ZScore'].abs() > 2).astype(int)
    
    # 订单流趋势
    result['OFI_Trend'] = result['OFI'].rolling(window=window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
    )
    
    # 成交量与订单流不平衡的相关性
    result['Volume_OFI_Correlation'] = (
        df[volume_col].rolling(window=window).corr(result['OFI'])
    )
    
    return result

def calculate_market_impact(df: pd.DataFrame,
                          price_col: str = 'Mid_Price',
                          volume_col: str = 'Volume',
                          ofi_col: str = 'OFI',
                          window: int = 20) -> pd.DataFrame:
    """
    计算市场冲击特征
    
    Args:
        df: 包含价格、成交量和订单流数据的DataFrame
        price_col: 价格列名
        volume_col: 成交量列名
        ofi_col: 订单流不平衡列名
        window: 计算窗口
    
    Returns:
        包含市场冲击特征的DataFrame
    """
    result = df.copy()
    
    # 价格冲击
    price_change = df[price_col].pct_change()
    result['Price_Impact'] = price_change
    
    # 临时价格冲击 (基于订单流不平衡)
    result['Temporary_Impact'] = result[ofi_col] * result['Price_Impact']
    
    # 永久价格冲击 (基于成交量)
    result['Permanent_Impact'] = (
        result['Price_Impact'] * df[volume_col]
    ).rolling(window=window).mean()
    
    # 市场冲击系数
    result['Impact_Coefficient'] = (
        result['Price_Impact'].abs() / df[volume_col]
    ).rolling(window=window).mean()
    
    # 订单流冲击
    result['Order_Flow_Impact'] = (
        result[ofi_col] * result['Price_Impact']
    ).rolling(window=window).mean()
    
    # 市场冲击的波动率
    result['Impact_Volatility'] = result['Price_Impact'].rolling(window=window).std()
    
    return result

def calculate_liquidity_features(df: pd.DataFrame,
                               bid_price_col: str = 'Bid_Price1',
                               ask_price_col: str = 'Ask_Price1',
                               bid_size_col: str = 'Bid_Size1',
                               ask_size_col: str = 'Ask_Size1',
                               volume_col: str = 'Volume',
                               window: int = 20) -> pd.DataFrame:
    """
    计算流动性特征
    
    Args:
        df: 包含订单簿和成交量数据的DataFrame
        bid_price_col: 买一价列名
        ask_price_col: 卖一价列名
        bid_size_col: 买一量列名
        ask_size_col: 卖一量列名
        volume_col: 成交量列名
        window: 计算窗口
    
    Returns:
        包含流动性特征的DataFrame
    """
    result = df.copy()
    
    # 买卖价差
    result['Spread'] = df[ask_price_col] - df[bid_price_col]
    result['Spread_Ratio'] = result['Spread'] / ((df[bid_price_col] + df[ask_price_col]) / 2)
    
    # 有效价差
    result['Effective_Spread'] = result['Spread'] * 2
    
    # 报价价差
    result['Quoted_Spread'] = result['Spread']
    
    # 相对价差
    result['Relative_Spread'] = result['Spread_Ratio']
    
    # 流动性比率
    result['Liquidity_Ratio'] = df[volume_col] / result['Spread']
    
    # 市场深度
    result['Market_Depth'] = df[bid_size_col] + df[ask_size_col]
    
    # 流动性指数
    result['Liquidity_Index'] = result['Market_Depth'] / result['Spread']
    
    # 流动性成本
    result['Liquidity_Cost'] = result['Spread'] / result['Market_Depth']
    
    # 流动性压力
    result['Liquidity_Pressure'] = (
        (df[ask_size_col] - df[bid_size_col]) / 
        (df[ask_size_col] + df[bid_size_col])
    )
    
    # 流动性趋势
    result['Liquidity_Trend'] = result['Liquidity_Index'].rolling(window=window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
    )
    
    return result

def calculate_order_book_features(df: pd.DataFrame,
                                levels: int = 5,
                                bid_price_prefix: str = 'Bid_Price',
                                bid_size_prefix: str = 'Bid_Size',
                                ask_price_prefix: str = 'Ask_Price',
                                ask_size_prefix: str = 'Ask_Size',
                                volume_col: str = 'Volume',
                                window: int = 20) -> pd.DataFrame:
    """
    计算综合订单簿特征
    
    Args:
        df: 包含订单簿数据的DataFrame
        levels: 订单簿档数
        bid_price_prefix: 买价列名前缀
        bid_size_prefix: 买量列名前缀
        ask_price_prefix: 卖价列名前缀
        ask_size_prefix: 卖量列名前缀
        volume_col: 成交量列名
        window: 计算窗口
    
    Returns:
        包含所有订单簿特征的DataFrame
    """
    result = df.copy()
    
    # 基础订单流不平衡特征
    result = calculate_order_flow_imbalance(result)
    
    # 多档订单簿不平衡特征
    result = calculate_order_book_imbalance(result, levels=levels,
                                          bid_price_prefix=bid_price_prefix,
                                          bid_size_prefix=bid_size_prefix,
                                          ask_price_prefix=ask_price_prefix,
                                          ask_size_prefix=ask_size_prefix)
    
    # 微观结构特征
    if 'Mid_Price' in result.columns and volume_col in result.columns:
        result = calculate_microstructure_features(result, price_col='Mid_Price', 
                                                 volume_col=volume_col, window=window)
    
    # 订单流指标
    if 'Bid_Size1' in result.columns and 'Ask_Size1' in result.columns:
        result = calculate_order_flow_metrics(result, window=window)
    
    # 市场冲击特征
    if 'Mid_Price' in result.columns and volume_col in result.columns and 'OFI' in result.columns:
        result = calculate_market_impact(result, price_col='Mid_Price', 
                                       volume_col=volume_col, ofi_col='OFI', window=window)
    
    # 流动性特征
    if all(col in result.columns for col in ['Bid_Price1', 'Ask_Price1', 'Bid_Size1', 'Ask_Size1']):
        result = calculate_liquidity_features(result, window=window)
    
    return result

def get_order_book_feature_summary(df: pd.DataFrame) -> dict:
    """
    获取订单簿特征摘要
    
    Args:
        df: 包含订单簿特征的DataFrame
    
    Returns:
        特征摘要字典
    """
    summary = {}
    
    # 订单流不平衡分析
    if 'OFI' in df.columns:
        ofi = df['OFI'].dropna()
        summary['订单流不平衡'] = {
            '当前值': ofi.iloc[-1] if len(ofi) > 0 else None,
            '平均值': ofi.mean(),
            '标准差': ofi.std(),
            '极值': {'最大值': ofi.max(), '最小值': ofi.min()}
        }
    
    # 价差分析
    if 'Spread_Ratio' in df.columns:
        spread_ratio = df['Spread_Ratio'].dropna()
        summary['价差'] = {
            '当前相对价差': spread_ratio.iloc[-1] if len(spread_ratio) > 0 else None,
            '平均相对价差': spread_ratio.mean(),
            '价差波动率': spread_ratio.std()
        }
    
    # 流动性分析
    if 'Liquidity_Index' in df.columns:
        liq_index = df['Liquidity_Index'].dropna()
        summary['流动性'] = {
            '当前流动性指数': liq_index.iloc[-1] if len(liq_index) > 0 else None,
            '平均流动性指数': liq_index.mean(),
            '流动性趋势': liq_index.iloc[-1] - liq_index.iloc[0] if len(liq_index) > 1 else None
        }
    
    # 市场冲击分析
    if 'Impact_Coefficient' in df.columns:
        impact_coef = df['Impact_Coefficient'].dropna()
        summary['市场冲击'] = {
            '当前冲击系数': impact_coef.iloc[-1] if len(impact_coef) > 0 else None,
            '平均冲击系数': impact_coef.mean(),
            '冲击系数变化': impact_coef.iloc[-1] - impact_coef.iloc[0] if len(impact_coef) > 1 else None
        }
    
    return summary 