#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本面特征计算模块
提供财务指标、估值指标、行业特征等基本面分析功能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FundamentalFeatures:
    """基本面特征计算器"""
    
    def __init__(self):
        """初始化基本面特征计算器"""
        self.feature_names = []
        
    def calculate_valuation_features(self, df: pd.DataFrame, 
                                   pe_data: Optional[pd.DataFrame] = None,
                                   pb_data: Optional[pd.DataFrame] = None,
                                   ps_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算估值特征
        
        Args:
            df: 基础数据DataFrame
            pe_data: 市盈率数据
            pb_data: 市净率数据
            ps_data: 市销率数据
            
        Returns:
            包含估值特征的DataFrame
        """
        result = df.copy()
        
        # 市盈率特征
        if pe_data is not None:
            result = self._calculate_pe_features(result, pe_data)
        
        # 市净率特征
        if pb_data is not None:
            result = self._calculate_pb_features(result, pb_data)
        
        # 市销率特征
        if ps_data is not None:
            result = self._calculate_ps_features(result, ps_data)
        
        # 估值分位数特征
        result = self._calculate_valuation_percentiles(result)
        
        return result
    
    def _calculate_pe_features(self, df: pd.DataFrame, pe_data: pd.DataFrame) -> pd.DataFrame:
        """计算市盈率特征"""
        result = df.copy()
        
        # 基础PE特征
        if 'PE' in pe_data.columns:
            result['PE'] = pe_data['PE']
            result['PE_MA_20'] = pe_data['PE'].rolling(20).mean()
            result['PE_MA_60'] = pe_data['PE'].rolling(60).mean()
            result['PE_Ratio'] = pe_data['PE'] / pe_data['PE'].rolling(20).mean()
            
            # PE分位数
            result['PE_Percentile_20'] = pe_data['PE'].rolling(252).rank(pct=True)
            result['PE_Percentile_60'] = pe_data['PE'].rolling(252).rank(pct=True)
            
            self.feature_names.extend(['PE', 'PE_MA_20', 'PE_MA_60', 'PE_Ratio', 
                                     'PE_Percentile_20', 'PE_Percentile_60'])
        
        return result
    
    def _calculate_pb_features(self, df: pd.DataFrame, pb_data: pd.DataFrame) -> pd.DataFrame:
        """计算市净率特征"""
        result = df.copy()
        
        if 'PB' in pb_data.columns:
            result['PB'] = pb_data['PB']
            result['PB_MA_20'] = pb_data['PB'].rolling(20).mean()
            result['PB_MA_60'] = pb_data['PB'].rolling(60).mean()
            result['PB_Ratio'] = pb_data['PB'] / pb_data['PB'].rolling(20).mean()
            
            # PB分位数
            result['PB_Percentile_20'] = pb_data['PB'].rolling(252).rank(pct=True)
            result['PB_Percentile_60'] = pb_data['PB'].rolling(252).rank(pct=True)
            
            self.feature_names.extend(['PB', 'PB_MA_20', 'PB_MA_60', 'PB_Ratio',
                                     'PB_Percentile_20', 'PB_Percentile_60'])
        
        return result
    
    def _calculate_ps_features(self, df: pd.DataFrame, ps_data: pd.DataFrame) -> pd.DataFrame:
        """计算市销率特征"""
        result = df.copy()
        
        if 'PS' in ps_data.columns:
            result['PS'] = ps_data['PS']
            result['PS_MA_20'] = ps_data['PS'].rolling(20).mean()
            result['PS_MA_60'] = ps_data['PS'].rolling(60).mean()
            result['PS_Ratio'] = ps_data['PS'] / ps_data['PS'].rolling(20).mean()
            
            # PS分位数
            result['PS_Percentile_20'] = ps_data['PS'].rolling(252).rank(pct=True)
            result['PS_Percentile_60'] = ps_data['PS'].rolling(252).rank(pct=True)
            
            self.feature_names.extend(['PS', 'PS_MA_20', 'PS_MA_60', 'PS_Ratio',
                                     'PS_Percentile_20', 'PS_Percentile_60'])
        
        return result
    
    def _calculate_valuation_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算估值分位数特征"""
        result = df.copy()
        
        # 综合估值分位数
        valuation_cols = [col for col in df.columns if any(x in col for x in ['PE', 'PB', 'PS'])]
        
        if len(valuation_cols) >= 2:
            # 计算估值综合分位数
            result['Valuation_Score'] = df[valuation_cols].rank(axis=1, pct=True).mean(axis=1)
            result['Valuation_Score_MA_20'] = result['Valuation_Score'].rolling(20).mean()
            
            self.feature_names.extend(['Valuation_Score', 'Valuation_Score_MA_20'])
        
        return result
    
    def calculate_financial_features(self, df: pd.DataFrame,
                                   revenue_data: Optional[pd.DataFrame] = None,
                                   profit_data: Optional[pd.DataFrame] = None,
                                   asset_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算财务特征
        
        Args:
            df: 基础数据DataFrame
            revenue_data: 营收数据
            profit_data: 利润数据
            asset_data: 资产数据
            
        Returns:
            包含财务特征的DataFrame
        """
        result = df.copy()
        
        # 营收增长特征
        if revenue_data is not None:
            result = self._calculate_revenue_features(result, revenue_data)
        
        # 利润增长特征
        if profit_data is not None:
            result = self._calculate_profit_features(result, profit_data)
        
        # 资产质量特征
        if asset_data is not None:
            result = self._calculate_asset_features(result, asset_data)
        
        return result
    
    def _calculate_revenue_features(self, df: pd.DataFrame, revenue_data: pd.DataFrame) -> pd.DataFrame:
        """计算营收特征"""
        result = df.copy()
        
        if 'Revenue' in revenue_data.columns:
            revenue = revenue_data['Revenue']
            
            # 营收增长率
            result['Revenue_Growth_YoY'] = revenue.pct_change(252)  # 年化增长率
            result['Revenue_Growth_QoQ'] = revenue.pct_change(63)   # 季度增长率
            
            # 营收趋势
            result['Revenue_MA_20'] = revenue.rolling(20).mean()
            result['Revenue_MA_60'] = revenue.rolling(60).mean()
            result['Revenue_Trend'] = (revenue - revenue.rolling(20).mean()) / revenue.rolling(20).std()
            
            # 营收稳定性
            result['Revenue_Volatility'] = revenue.rolling(60).std() / revenue.rolling(60).mean()
            
            self.feature_names.extend(['Revenue_Growth_YoY', 'Revenue_Growth_QoQ', 
                                     'Revenue_MA_20', 'Revenue_MA_60', 'Revenue_Trend',
                                     'Revenue_Volatility'])
        
        return result
    
    def _calculate_profit_features(self, df: pd.DataFrame, profit_data: pd.DataFrame) -> pd.DataFrame:
        """计算利润特征"""
        result = df.copy()
        
        if 'NetProfit' in profit_data.columns:
            profit = profit_data['NetProfit']
            
            # 利润增长率
            result['Profit_Growth_YoY'] = profit.pct_change(252)
            result['Profit_Growth_QoQ'] = profit.pct_change(63)
            
            # 利润率
            if 'Revenue' in profit_data.columns:
                result['Profit_Margin'] = profit / profit_data['Revenue']
                result['Profit_Margin_MA_20'] = result['Profit_Margin'].rolling(20).mean()
            
            # 利润趋势
            result['Profit_MA_20'] = profit.rolling(20).mean()
            result['Profit_MA_60'] = profit.rolling(60).mean()
            result['Profit_Trend'] = (profit - profit.rolling(20).mean()) / profit.rolling(20).std()
            
            self.feature_names.extend(['Profit_Growth_YoY', 'Profit_Growth_QoQ',
                                     'Profit_Margin', 'Profit_Margin_MA_20',
                                     'Profit_MA_20', 'Profit_MA_60', 'Profit_Trend'])
        
        return result
    
    def _calculate_asset_features(self, df: pd.DataFrame, asset_data: pd.DataFrame) -> pd.DataFrame:
        """计算资产特征"""
        result = df.copy()
        
        if 'TotalAssets' in asset_data.columns:
            assets = asset_data['TotalAssets']
            
            # 资产增长率
            result['Asset_Growth_YoY'] = assets.pct_change(252)
            result['Asset_Growth_QoQ'] = assets.pct_change(63)
            
            # 资产周转率
            if 'Revenue' in asset_data.columns:
                result['Asset_Turnover'] = asset_data['Revenue'] / assets
                result['Asset_Turnover_MA_20'] = result['Asset_Turnover'].rolling(20).mean()
            
            # ROA
            if 'NetProfit' in asset_data.columns:
                result['ROA'] = asset_data['NetProfit'] / assets
                result['ROA_MA_20'] = result['ROA'].rolling(20).mean()
            
            self.feature_names.extend(['Asset_Growth_YoY', 'Asset_Growth_QoQ',
                                     'Asset_Turnover', 'Asset_Turnover_MA_20',
                                     'ROA', 'ROA_MA_20'])
        
        return result
    
    def calculate_industry_features(self, df: pd.DataFrame,
                                  industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算行业特征
        
        Args:
            df: 基础数据DataFrame
            industry_data: 行业数据
            
        Returns:
            包含行业特征的DataFrame
        """
        result = df.copy()
        
        if industry_data is not None:
            # 行业相对表现
            if 'Industry_Return' in industry_data.columns:
                result['Industry_Relative_Return'] = (
                    result['ClosePrice'].pct_change() - industry_data['Industry_Return']
                )
                result['Industry_Relative_Return_MA_20'] = (
                    result['Industry_Relative_Return'].rolling(20).mean()
                )
            
            # 行业估值相对水平
            if 'Industry_PE' in industry_data.columns:
                result['Industry_PE_Ratio'] = result['PE'] / industry_data['Industry_PE']
                result['Industry_PE_Ratio_MA_20'] = result['Industry_PE_Ratio'].rolling(20).mean()
            
            # 行业资金流向
            if 'Industry_Volume' in industry_data.columns:
                result['Industry_Volume_Ratio'] = (
                    result['TurnoverVolume'] / industry_data['Industry_Volume']
                )
            
            self.feature_names.extend(['Industry_Relative_Return', 'Industry_Relative_Return_MA_20',
                                     'Industry_PE_Ratio', 'Industry_PE_Ratio_MA_20',
                                     'Industry_Volume_Ratio'])
        
        return result
    
    def calculate_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算质量特征
        
        Args:
            df: 基础数据DataFrame
            
        Returns:
            包含质量特征的DataFrame
        """
        result = df.copy()
        
        # 数据质量指标
        result['Data_Quality_Score'] = 1.0  # 基础分数
        
        # 检查价格数据质量
        if 'ClosePrice' in result.columns:
            # 价格连续性
            price_changes = result['ClosePrice'].pct_change().abs()
            result['Price_Continuity'] = (price_changes < 0.2).rolling(20).mean()
            
            # 价格合理性
            result['Price_Reasonableness'] = (
                (result['ClosePrice'] > 0) & 
                (result['ClosePrice'] < 10000)
            ).astype(float)
        
        # 检查成交量数据质量
        if 'TurnoverVolume' in result.columns:
            # 成交量合理性
            result['Volume_Reasonableness'] = (
                (result['TurnoverVolume'] > 0) & 
                (result['TurnoverVolume'] < 1e12)
            ).astype(float)
        
        # 综合质量分数
        quality_cols = [col for col in result.columns if 'Quality' in col or 'Reasonableness' in col]
        if quality_cols:
            result['Overall_Quality_Score'] = result[quality_cols].mean(axis=1)
        
        self.feature_names.extend(['Data_Quality_Score', 'Price_Continuity', 
                                 'Price_Reasonableness', 'Volume_Reasonableness',
                                 'Overall_Quality_Score'])
        
        return result
    
    def calculate_all_fundamental_features(self, df: pd.DataFrame,
                                         **kwargs) -> pd.DataFrame:
        """
        计算所有基本面特征
        
        Args:
            df: 基础数据DataFrame
            **kwargs: 其他数据参数
            
        Returns:
            包含所有基本面特征的DataFrame
        """
        logger.info("开始计算基本面特征...")
        
        result = df.copy()
        
        # 计算各类基本面特征
        result = self.calculate_valuation_features(result, **kwargs)
        result = self.calculate_financial_features(result, **kwargs)
        result = self.calculate_industry_features(result, **kwargs)
        result = self.calculate_quality_features(result)
        
        logger.info(f"基本面特征计算完成，共生成 {len(self.feature_names)} 个特征")
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """获取所有生成的特征名称"""
        return self.feature_names.copy()


if __name__ == "__main__":
    # 创建模拟基本面数据
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='Q')
    
    financial_data = []
    for symbol in symbols:
        for date in dates:
            financial_data.append({
                'symbol': symbol,
                'date': date,
                'eps': np.random.uniform(1, 5),
                'total_revenue': np.random.uniform(1e9, 1e11),
                'net_income': np.random.uniform(1e8, 1e10),
                'total_assets': np.random.uniform(1e10, 1e12),
                'total_equity': np.random.uniform(1e9, 1e11),
                'total_debt': np.random.uniform(1e8, 1e10),
                'cash': np.random.uniform(1e8, 1e10),
                'ebitda': np.random.uniform(1e8, 1e10),
                'cogs': np.random.uniform(1e8, 1e9),
                'operating_income': np.random.uniform(1e8, 1e9),
                'dividend_per_share': np.random.uniform(0.5, 2.0),
                'shares_outstanding': np.random.uniform(1e9, 1e10),
                'current_assets': np.random.uniform(1e9, 1e10),
                'current_liabilities': np.random.uniform(1e8, 1e9),
                'inventory': np.random.uniform(1e7, 1e8),
                'ebit': np.random.uniform(1e8, 1e9),
                'interest_expense': np.random.uniform(1e6, 1e7),
                'market_cap': np.random.uniform(1e11, 1e12)
            })
    
    financials_df = pd.DataFrame(financial_data)
    
    # 创建模拟股价数据
    price_data = []
    for symbol in symbols:
        for date in dates:
            price_data.append({
                'symbol': symbol,
                'date': date,
                'close': np.random.uniform(100, 300)
            })
    
    prices_df = pd.DataFrame(price_data)
    
    # 创建模拟分析师数据
    analyst_data = []
    ratings = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
    for symbol in symbols:
        for date in dates:
            analyst_data.append({
                'symbol': symbol,
                'date': date,
                'consensus_eps': np.random.uniform(1, 5),
                'actual_eps': np.random.uniform(0.8, 1.2) * np.random.uniform(1, 5),
                'rating': np.random.choice(ratings)
            })
    
    analyst_df = pd.DataFrame(analyst_data)
    
    # 计算基本面特征
    ff = FundamentalFeatures()
    features_df = ff.calculate_all_fundamental_features(financials_df, prices_df=prices_df, analyst_df=analyst_df)
    
    # 保存结果
    features_df.to_csv('fundamental_features.csv', index=False)
    
    # 打印结果
    print("基本面特征计算完成!")
    print("特征列:", features_df.columns.tolist())
    print("示例数据:")
    print(features_df.head())
    
    # 可视化特征分布
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(15, 10))
    
    # 估值比率
    plt.subplot(2, 2, 1)
    sns.boxplot(data=features_df, y='P/E')
    plt.title('P/E分布')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(data=features_df, y='P/B')
    plt.title('P/B分布')
    
    # 盈利能力
    plt.subplot(2, 2, 3)
    sns.lineplot(data=features_df, x='date', y='ROE', hue='symbol')
    plt.title('ROE变化趋势')
    
    plt.subplot(2, 2, 4)
    sns.lineplot(data=features_df, x='date', y='Gross_Margin', hue='symbol')
    plt.title('毛利率变化趋势')
    
    plt.tight_layout()
    plt.savefig('fundamental_features_visualization.png')
    plt.show()