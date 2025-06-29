#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复权因子补全工具
用于为股票行情数据补充精确的复权因子
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connector import JuyuanDB

class AdjustingFactorFiller:
    """复权因子补全器"""
    
    def __init__(self):
        self.db = None
    
    def connect_db(self):
        """连接数据库"""
        try:
            self.db = JuyuanDB()
            return True
        except Exception as e:
            logging.error(f"数据库连接失败: {str(e)}")
            return False
    
    def close_db(self):
        """关闭数据库连接"""
        if self.db:
            try:
                self.db.close()
            except Exception as e:
                logging.error(f"关闭数据库连接失败: {str(e)}")
    
    def get_adjusting_factors(self, end_date):
        """获取复权因子数据
        
        Args:
            end_date: 结束日期，格式：'YYYY-MM-DD'
            
        Returns:
            DataFrame: 包含InnerCode, ExDiviDate, RatioAdjustingFactor的复权因子数据
        """
        if not self.db:
            logging.error("数据库未连接")
            return pd.DataFrame()
        
        sql = f"""
        SELECT 
            InnerCode,
            ExDiviDate,
            RatioAdjustingFactor
        FROM QT_AdjustingFactor
        WHERE ExDiviDate <= '{end_date}'
        ORDER BY InnerCode, ExDiviDate
        """
        
        try:
            adj_factors = self.db.read_sql(sql)
            logging.info(f"获取复权因子数据成功，共 {len(adj_factors)} 条记录")
            return adj_factors
        except Exception as e:
            logging.error(f"获取复权因子数据失败: {str(e)}")
            return pd.DataFrame()
    
    def fill_adjusting_factors(self, df, adj_factors):
        """为行情数据补充复权因子
        
        Args:
            df: 行情数据DataFrame，必须包含InnerCode和TradingDay字段
            adj_factors: 复权因子数据DataFrame，包含InnerCode, ExDiviDate, RatioAdjustingFactor
            
        Returns:
            DataFrame: 添加了AdjustingFactor字段的行情数据
        """
        if df.empty:
            logging.warning("行情数据为空")
            return df
        
        if adj_factors.empty:
            logging.warning("复权因子数据为空，所有复权因子设为1.0")
            df['AdjustingFactor'] = 1.0
            return df
        
        # 检查必要字段
        required_fields = ['InnerCode', 'TradingDay']
        for field in required_fields:
            if field not in df.columns:
                logging.error(f"行情数据缺少必要字段: {field}")
                return df
        
        # 确保日期格式正确
        df['TradingDay'] = pd.to_datetime(df['TradingDay'])
        adj_factors['ExDiviDate'] = pd.to_datetime(adj_factors['ExDiviDate'])
        
        # 初始化复权因子为1.0
        df['AdjustingFactor'] = 1.0
        
        # 创建复权因子的副本，用于合并
        adj_factors_copy = adj_factors.copy()
        adj_factors_copy = adj_factors_copy.rename(columns={'RatioAdjustingFactor': 'AdjustingFactor'})
        
        # 使用merge_asof进行高效的向前填充
        try:
            # 按InnerCode分组，对每个交易日向前填充最近的复权因子
            result_df = pd.merge_asof(
                df.sort_values(['InnerCode', 'TradingDay']),
                adj_factors_copy.sort_values(['InnerCode', 'ExDiviDate']),
                left_on='TradingDay',
                right_on='ExDiviDate',
                by='InnerCode',
                direction='backward'  # 向前填充
            )
            
            # 处理没有复权因子的情况，设为1.0
            result_df['AdjustingFactor'] = result_df['AdjustingFactor'].fillna(1.0)
            
            # 删除多余的ExDiviDate列
            if 'ExDiviDate' in result_df.columns:
                result_df = result_df.drop('ExDiviDate', axis=1)
            
            logging.info(f"复权因子补全完成，共处理 {len(result_df)} 行数据")
            return result_df
            
        except Exception as e:
            logging.warning(f"merge_asof失败，使用备用方法: {str(e)}")
            return self._fill_adjusting_factors_fallback(df, adj_factors)
    
    def _fill_adjusting_factors_fallback(self, df, adj_factors):
        """备用方法：使用循环进行复权因子补全"""
        logging.info("使用备用方法补全复权因子...")
        
        # 按股票代码分组处理
        for inner_code in df['InnerCode'].unique():
            # 获取该股票的所有复权因子记录
            stock_adj_factors = adj_factors[adj_factors['InnerCode'] == inner_code].copy()
            
            if stock_adj_factors.empty:
                continue
            
            # 获取该股票的所有交易日
            stock_mask = df['InnerCode'] == inner_code
            stock_df = df[stock_mask].copy()
            
            # 为每个交易日找到最近的复权因子
            for idx in stock_df.index:
                trading_day = stock_df.loc[idx, 'TradingDay']
                
                # 找到该交易日及之前最近的复权因子
                valid_factors = stock_adj_factors[stock_adj_factors['ExDiviDate'] <= trading_day]
                
                if not valid_factors.empty:
                    # 取最近的复权因子
                    latest_factor = valid_factors.loc[valid_factors['ExDiviDate'].idxmax(), 'RatioAdjustingFactor']
                    df.loc[idx, 'AdjustingFactor'] = latest_factor
        
        logging.info(f"复权因子补全完成（备用方法），共处理 {len(df)} 行数据")
        return df
    
    def validate_adjusting_factors(self, df):
        """验证复权因子的合理性
        
        Args:
            df: 包含AdjustingFactor字段的DataFrame
            
        Returns:
            dict: 验证结果统计
        """
        if 'AdjustingFactor' not in df.columns:
            return {'error': '缺少AdjustingFactor字段'}
        
        validation_results = {
            'total_rows': len(df),
            'null_count': df['AdjustingFactor'].isna().sum(),
            'zero_count': (df['AdjustingFactor'] == 0).sum(),
            'negative_count': (df['AdjustingFactor'] < 0).sum(),
            'too_small_count': (df['AdjustingFactor'] < 0.001).sum(),
            'too_large_count': (df['AdjustingFactor'] > 1000).sum(),
            'normal_count': ((df['AdjustingFactor'] >= 0.001) & (df['AdjustingFactor'] <= 1000)).sum(),
            'min_value': df['AdjustingFactor'].min(),
            'max_value': df['AdjustingFactor'].max(),
            'mean_value': df['AdjustingFactor'].mean(),
            'median_value': df['AdjustingFactor'].median()
        }
        
        return validation_results


def main():
    """主函数 - 示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='复权因子补全工具')
    parser.add_argument('--start', default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', default='output/adjusting_factors_sample.csv', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建补全器
    filler = AdjustingFactorFiller()
    
    if not filler.connect_db():
        return
    
    try:
        # 获取复权因子数据
        adj_factors = filler.get_adjusting_factors(args.end)
        
        if adj_factors.empty:
            logging.error("无法获取复权因子数据")
            return
        
        # 获取示例行情数据
        sample_sql = f"""
        SELECT 
            a.InnerCode,
            c.SecuCode,
            a.TradingDay,
            a.ClosePrice
        FROM QT_DailyQuote a
        LEFT JOIN SecuMain c ON a.InnerCode = c.InnerCode
        WHERE c.SecuCategory = 1 
            AND c.SecuMarket IN (83, 90) 
            AND c.ListedState = 1
            AND a.TradingDay BETWEEN '{args.start}' AND '{args.end}'
        ORDER BY a.TradingDay, c.SecuCode
        LIMIT 1000
        """
        
        sample_data = filler.db.read_sql(sample_sql)
        
        if sample_data.empty:
            logging.error("无法获取示例行情数据")
            return
        
        # 补全复权因子
        result_df = filler.fill_adjusting_factors(sample_data, adj_factors)
        
        # 验证结果
        validation = filler.validate_adjusting_factors(result_df)
        
        print("\n=== 复权因子验证结果 ===")
        for key, value in validation.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # 保存结果
        result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {args.output}")
        
    finally:
        filler.close_db()


if __name__ == '__main__':
    main() 