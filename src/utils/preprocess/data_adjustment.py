#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据调整模块
包含价格复权、数据清洗等预处理功能
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger('data_adjustment')

class DataAdjustment:
    """数据调整处理器"""
    
    def __init__(self):
        """初始化处理器"""
        self.logger = logging.getLogger(__name__)
        self.price_columns = ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'PrevClosePrice']
        self.volume_columns = ['TurnoverVolume', 'TurnoverValue']
    
    def adjust_prices(
        self, 
        df: pd.DataFrame, 
        adj_type: str = 'forward',
        adj_factor_col: str = 'AdjFactor'
    ) -> pd.DataFrame:
        """
        价格复权处理
        
        Args:
            df: 包含价格数据的DataFrame
            adj_type: 复权类型 ('forward', 'backward', 'none')
            adj_factor_col: 复权因子列名
            
        Returns:
            pd.DataFrame: 复权后的数据
        """
        if adj_type == 'none' or adj_factor_col not in df.columns:
            return df.copy()
        
        result_df = df.copy()
        
        # 按股票代码分组处理
        for secu_code in result_df['SecuCode'].unique():
            mask = result_df['SecuCode'] == secu_code
            stock_data = result_df[mask].copy()
            
            if adj_factor_col in stock_data.columns:
                adj_factors = stock_data[adj_factor_col].fillna(1.0)
                
                # 前复权：以当前价格为基准
                if adj_type == 'forward':
                    for col in self.price_columns:
                        if col in stock_data.columns:
                            stock_data[f'{col}_Adj'] = stock_data[col] * adj_factors
                
                # 后复权：以历史价格为基准
                elif adj_type == 'backward':
                    for col in self.price_columns:
                        if col in stock_data.columns:
                            stock_data[f'{col}_Adj'] = stock_data[col] / adj_factors
                
                result_df.loc[mask] = stock_data
        
        self.logger.info(f"完成{adj_type}复权处理")
        return result_df
    
    def clean_outliers(
        self, 
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        异常值清洗
        
        Args:
            df: 输入数据DataFrame
            columns: 要清洗的列名列表
            method: 异常值检测方法 ('iqr', 'zscore', 'mad')
            threshold: 阈值倍数
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        result_df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in result_df.columns:
                if method == 'iqr':
                    # IQR方法
                    Q1 = result_df[col].quantile(0.25)
                    Q3 = result_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # 将异常值设为NaN
                    result_df.loc[(result_df[col] < lower_bound) | 
                                 (result_df[col] > upper_bound), col] = np.nan
                
                elif method == 'zscore':
                    # Z-score方法
                    z_scores = np.abs((result_df[col] - result_df[col].mean()) / result_df[col].std())
                    result_df.loc[z_scores > threshold, col] = np.nan
                
                elif method == 'mad':
                    # MAD方法
                    median = result_df[col].median()
                    mad = np.median(np.abs(result_df[col] - median))
                    modified_z_scores = 0.6745 * (result_df[col] - median) / mad
                    result_df.loc[np.abs(modified_z_scores) > threshold, col] = np.nan
        
        self.logger.info(f"异常值清洗完成，方法: {method}")
        return result_df
    
    def fill_missing_values(
        self, 
        df: pd.DataFrame,
        method: str = 'ffill',
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        缺失值填充
        
        Args:
            df: 输入数据DataFrame
            method: 填充方法 ('ffill', 'bfill', 'interpolate', 'mean', 'median')
            limit: 填充限制
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        result_df = df.copy()
        
        if method in ['ffill', 'bfill']:
            result_df = result_df.fillna(method=method, limit=limit)
        elif method == 'interpolate':
            result_df = result_df.interpolate(limit=limit)
        elif method in ['mean', 'median']:
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if method == 'mean':
                    fill_value = result_df[col].mean()
                else:
                    fill_value = result_df[col].median()
                result_df[col].fillna(fill_value, inplace=True)
        
        self.logger.info(f"缺失值填充完成，方法: {method}")
        return result_df
    
    def normalize_data(
        self, 
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        数据标准化
        
        Args:
            df: 输入数据DataFrame
            columns: 要标准化的列名列表
            method: 标准化方法 ('zscore', 'minmax', 'robust')
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        result_df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in result_df.columns:
                if method == 'zscore':
                    mean_val = result_df[col].mean()
                    std_val = result_df[col].std()
                    result_df[f'{col}_normalized'] = (result_df[col] - mean_val) / std_val
                
                elif method == 'minmax':
                    min_val = result_df[col].min()
                    max_val = result_df[col].max()
                    result_df[f'{col}_normalized'] = (result_df[col] - min_val) / (max_val - min_val)
                
                elif method == 'robust':
                    median_val = result_df[col].median()
                    mad_val = np.median(np.abs(result_df[col] - median_val))
                    result_df[f'{col}_normalized'] = (result_df[col] - median_val) / mad_val
        
        self.logger.info(f"数据标准化完成，方法: {method}")
        return result_df
    
    def filter_suspension_days(
        self, 
        df: pd.DataFrame,
        suspension_col: str = 'Ifsuspend'
    ) -> pd.DataFrame:
        """
        过滤停牌日
        
        Args:
            df: 输入数据DataFrame
            suspension_col: 停牌标识列名
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        result_df = df.copy()
        
        if suspension_col in result_df.columns:
            # 过滤停牌股票
            result_df = result_df[
                (result_df[suspension_col].isna()) | (result_df[suspension_col] == 0)
            ]
        
        self.logger.info(f"停牌日过滤完成，剩余数据量: {len(result_df)}")
        return result_df
    
    def validate_data_quality(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        数据质量验证
        
        Args:
            df: 输入数据DataFrame
            
        Returns:
            Dict: 数据质量报告
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # 检查缺失值
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                quality_report['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': missing_count / len(df) * 100
                }
        
        # 检查数值列的统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        quality_report['numeric_stats'] = {}
        
        for col in numeric_cols:
            quality_report['numeric_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
        
        self.logger.info("数据质量验证完成")
        return quality_report 