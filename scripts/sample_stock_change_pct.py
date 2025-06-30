#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票行情历史表随机采样脚本
每月随机采样1000条数据，使用12个线程并发处理
分析涨跌幅（ChangePCT）的统计特征
检查单日涨幅超过8%的数据是否发生除权除息
关联停牌复牌信息，分析复牌日涨幅变化
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connector import JuyuanDB

class StockQuoteSampler:
    """股票行情历史表随机采样器"""
    
    def __init__(self, year=2024, sample_size=1000, max_workers=12):
        self.year = year
        self.sample_size = sample_size
        self.max_workers = max_workers
        self.output_dir = Path('output/processed_data/stock_quote_samples')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 添加线程锁用于保护输出
        self.print_lock = threading.Lock()
        
    def safe_print(self, message):
        """线程安全的打印函数"""
        with self.print_lock:
            print(message)
    
    def check_suspend_resumption(self, df, db):
        """检查停牌复牌信息"""
        if df.empty:
            return df
        
        # 获取所有涉及到的股票代码和日期
        stock_dates = df[['SecuCode', 'TradingDay']].drop_duplicates()
        
        if stock_dates.empty:
            return df
        
        # 构建查询停牌复牌信息的SQL
        stock_codes = "','".join(stock_dates['SecuCode'].unique())
        trading_dates = "','".join(stock_dates['TradingDay'].dt.strftime('%Y-%m-%d').unique())
        
        sql = f"""
        SELECT 
            c.SecuCode,
            s.SuspendDate,
            s.SuspendTime,
            s.SuspendReason,
            s.SuspendStatement,
            s.SuspendTerm,
            s.SuspendType,
            s.ResumptionDate,
            s.ResumptionTime,
            s.ResumptionStatement,
            s.InfoPublDate,
            s.InfoSource
        FROM LC_SuspendResumption s
        LEFT JOIN SecuMain c ON s.InnerCode = c.InnerCode
        WHERE c.SecuCode IN ('{stock_codes}')
            AND (s.SuspendDate IN ('{trading_dates}') OR s.ResumptionDate IN ('{trading_dates}'))
        ORDER BY c.SecuCode, s.SuspendDate, s.ResumptionDate
        """
        
        try:
            suspend_resumption = db.read_sql(sql)
            
            if not suspend_resumption.empty:
                # 标记停牌日
                suspend_data = suspend_resumption[suspend_resumption['SuspendDate'].notna()].copy()
                if not suspend_data.empty:
                    df = df.merge(
                        suspend_data[['SecuCode', 'SuspendDate', 'SuspendReason', 'SuspendStatement', 'SuspendTerm']],
                        left_on=['SecuCode', 'TradingDay'],
                        right_on=['SecuCode', 'SuspendDate'],
                        how='left',
                        suffixes=('', '_suspend')
                    )
                    df['IsSuspendDay'] = df['SuspendDate'].notna()
                else:
                    df['IsSuspendDay'] = False
                
                # 标记复牌日
                resumption_data = suspend_resumption[suspend_resumption['ResumptionDate'].notna()].copy()
                if not resumption_data.empty:
                    df = df.merge(
                        resumption_data[['SecuCode', 'ResumptionDate', 'ResumptionStatement', 'SuspendReason', 'SuspendTerm']],
                        left_on=['SecuCode', 'TradingDay'],
                        right_on=['SecuCode', 'ResumptionDate'],
                        how='left',
                        suffixes=('', '_resumption')
                    )
                    df['IsResumptionDay'] = df['ResumptionDate'].notna()
                else:
                    df['IsResumptionDay'] = False
                
                self.safe_print(f"找到 {len(suspend_resumption)} 条停牌复牌记录")
            else:
                df['IsSuspendDay'] = False
                df['IsResumptionDay'] = False
                self.safe_print("未找到停牌复牌记录")
                
        except Exception as e:
            self.safe_print(f"查询停牌复牌信息时出错: {str(e)}")
            df['IsSuspendDay'] = False
            df['IsResumptionDay'] = False
        
        return df
    
    def check_adjusting_factors(self, df, db):
        """检查除权除息因子"""
        if df.empty:
            return df
        
        # 获取所有涉及到的股票代码和日期
        stock_dates = df[['SecuCode', 'TradingDay']].drop_duplicates()
        
        if stock_dates.empty:
            return df
        
        # 构建查询除权除息因子的SQL
        stock_codes = "','".join(stock_dates['SecuCode'].unique())
        trading_dates = "','".join(stock_dates['TradingDay'].dt.strftime('%Y-%m-%d').unique())
        
        sql = f"""
        SELECT 
            c.SecuCode,
            a.ExDiviDate,
            a.AdjustingFactor,
            a.AdjustingConst,
            a.RatioAdjustingFactor,
            a.AccuCashDivi,
            a.AccuBonusShareRatio
        FROM QT_AdjustingFactor a
        LEFT JOIN SecuMain c ON a.InnerCode = c.InnerCode
        WHERE c.SecuCode IN ('{stock_codes}')
            AND a.ExDiviDate IN ('{trading_dates}')
        ORDER BY c.SecuCode, a.ExDiviDate
        """
        
        try:
            adjusting_factors = db.read_sql(sql)
            
            if not adjusting_factors.empty:
                # 合并除权除息信息到原数据
                df = df.merge(
                    adjusting_factors,
                    left_on=['SecuCode', 'TradingDay'],
                    right_on=['SecuCode', 'ExDiviDate'],
                    how='left'
                )
                
                # 标记是否有除权除息
                df['HasAdjustingFactor'] = df['AdjustingFactor'].notna()
                
                self.safe_print(f"找到 {len(adjusting_factors)} 条除权除息记录")
            else:
                df['HasAdjustingFactor'] = False
                self.safe_print("未找到除权除息记录")
                
        except Exception as e:
            self.safe_print(f"查询除权除息因子时出错: {str(e)}")
            df['HasAdjustingFactor'] = False
        
        return df
    
    def analyze_change_pct(self, df):
        """分析涨跌幅的统计特征"""
        if df.empty:
            return {}
        
        # 过滤有效的涨跌幅数据
        valid_data = df.dropna(subset=['ChangePCT'])
        
        if valid_data.empty:
            return {}
        
        # 基本统计信息
        stats = {
            'total_records': len(df),
            'valid_records': len(valid_data),
            'change_pct_stats': {
                'mean': valid_data['ChangePCT'].mean(),
                'median': valid_data['ChangePCT'].median(),
                'std': valid_data['ChangePCT'].std(),
                'min': valid_data['ChangePCT'].min(),
                'max': valid_data['ChangePCT'].max(),
                'q25': valid_data['ChangePCT'].quantile(0.25),
                'q75': valid_data['ChangePCT'].quantile(0.75),
                'positive_count': (valid_data['ChangePCT'] > 0).sum(),
                'negative_count': (valid_data['ChangePCT'] < 0).sum(),
                'zero_count': (valid_data['ChangePCT'] == 0).sum(),
                'limit_up_count': (valid_data['ChangePCT'] >= 9.8).sum(),  # 涨停
                'limit_down_count': (valid_data['ChangePCT'] <= -9.8).sum(),  # 跌停
                'high_change_count': (valid_data['ChangePCT'] >= 8.0).sum(),  # 涨幅超过8%
                'small_change_count': ((valid_data['ChangePCT'] >= -1) & (valid_data['ChangePCT'] <= 1)).sum(),  # 小幅波动
                'medium_change_count': (((valid_data['ChangePCT'] > 1) & (valid_data['ChangePCT'] <= 5)) | 
                                       ((valid_data['ChangePCT'] < -1) & (valid_data['ChangePCT'] >= -5))).sum(),  # 中等波动
                'large_change_count': (((valid_data['ChangePCT'] > 5) & (valid_data['ChangePCT'] < 9.8)) | 
                                      ((valid_data['ChangePCT'] < -5) & (valid_data['ChangePCT'] > -9.8))).sum()  # 大幅波动
            }
        }
        
        # 复牌日统计
        if 'IsResumptionDay' in valid_data.columns:
            resumption_data = valid_data[valid_data['IsResumptionDay'] == True]
            if not resumption_data.empty:
                stats['resumption_stats'] = {
                    'count': len(resumption_data),
                    'mean': resumption_data['ChangePCT'].mean(),
                    'median': resumption_data['ChangePCT'].median(),
                    'std': resumption_data['ChangePCT'].std(),
                    'min': resumption_data['ChangePCT'].min(),
                    'max': resumption_data['ChangePCT'].max(),
                    'positive_count': (resumption_data['ChangePCT'] > 0).sum(),
                    'negative_count': (resumption_data['ChangePCT'] < 0).sum(),
                    'limit_up_count': (resumption_data['ChangePCT'] >= 9.8).sum(),
                    'limit_down_count': (resumption_data['ChangePCT'] <= -9.8).sum(),
                    'high_change_count': (resumption_data['ChangePCT'] >= 8.0).sum()
                }
        
        return stats
        
    def generate_monthly_ranges(self):
        """生成2024年各月的时间范围"""
        ranges = []
        
        for month in range(1, 13):
            start_date = datetime(self.year, month, 1)
            
            # 计算当月结束日期
            if month == 12:
                end_date = datetime(self.year, month, 31)
            else:
                next_month = datetime(self.year, month + 1, 1)
                end_date = next_month - timedelta(days=1)
            
            ranges.append({
                'year': self.year,
                'month': month,
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            })
        
        return ranges
    
    def sample_month_data(self, month_range):
        """采样单个月份的数据"""
        year = month_range['year']
        month = month_range['month']
        start = month_range['start']
        end = month_range['end']
        
        self.safe_print(f"开始采样 {year}年{month}月 数据 ({start} 到 {end})")
        
        conn = None
        try:
            # 创建数据库连接
            db = JuyuanDB()
            
            # 构建查询SQL
            q_key = ['TradingDay', 'PrevClosePrice', 'OpenPrice', 'HighPrice', 
                    'LowPrice', 'ClosePrice', 'TurnoverVolume', 'TurnoverValue']
            l_key = ['PriceCeiling', 'PriceFloor']
            
            q_key_str = ','.join(['a.' + s for s in q_key])
            l_key_str = ','.join(['b.' + s for s in l_key])
            
            sql = f"""
            SELECT 
                c.SecuCode,
                c.SecuMarket,
                d.Ifsuspend,
                d.ChangePCT,
                {q_key_str},
                {l_key_str}
            FROM QT_DailyQuote a
            LEFT JOIN QT_PriceLimit b
                ON a.InnerCode = b.InnerCode AND a.TradingDay = b.TradingDay
            LEFT JOIN QT_StockPerformance d
                ON a.InnerCode = d.InnerCode AND a.TradingDay = d.TradingDay
            LEFT JOIN SecuMain c
                ON a.InnerCode = c.InnerCode
            WHERE (c.SecuCategory = 1 AND c.SecuMarket IN (83, 90) AND c.ListedState = 1)
                AND a.TradingDay BETWEEN '{start}' AND '{end}'
            ORDER BY a.TradingDay, c.SecuCode
            """
            
            # 执行查询
            df = db.read_sql(sql)
            
            if df.empty:
                self.safe_print(f"{year}年{month}月: 无数据")
                return {
                    'year': year,
                    'month': month,
                    'total_rows': 0,
                    'sampled_rows': 0,
                    'filename': None,
                    'stats': {},
                    'high_change_filename': None,
                    'resumption_filename': None
                }
            
            # 检查停牌复牌信息
            df = self.check_suspend_resumption(df, db)
            
            # 检查除权除息因子
            df = self.check_adjusting_factors(df, db)
            
            # 分析涨跌幅统计
            stats = self.analyze_change_pct(df)
            
            # 筛选涨幅超过8%的数据
            high_change_data = df[df['ChangePCT'] >= 8.0].copy()
            
            # 保存涨幅超过8%的数据
            if not high_change_data.empty:
                high_change_filename = self.output_dir / f"{year}_{month:02d}_high_change_8pct.csv"
                high_change_data.to_csv(high_change_filename, index=False, encoding='utf-8-sig')
                self.safe_print(f"涨幅超过8%的数据已保存到: {high_change_filename} (共{len(high_change_data)}条)")
                
                # 统计除权除息情况
                adjusting_count = high_change_data['HasAdjustingFactor'].sum()
                self.safe_print(f"涨幅超过8%的数据中，有{adjusting_count}条发生除权除息")
            else:
                high_change_filename = None
                self.safe_print(f"{year}年{month}月: 无涨幅超过8%的数据")
            
            # 筛选复牌日数据
            resumption_data = df[df['IsResumptionDay'] == True].copy()
            
            # 保存复牌日数据
            if not resumption_data.empty:
                resumption_filename = self.output_dir / f"{year}_{month:02d}_resumption_days.csv"
                resumption_data.to_csv(resumption_filename, index=False, encoding='utf-8-sig')
                self.safe_print(f"复牌日数据已保存到: {resumption_filename} (共{len(resumption_data)}条)")
                
                # 统计复牌日涨跌幅
                if 'resumption_stats' in stats:
                    resumption_stats = stats['resumption_stats']
                    self.safe_print(f"复牌日涨跌幅统计:")
                    self.safe_print(f"  复牌日数量: {resumption_stats['count']}")
                    self.safe_print(f"  平均涨幅: {resumption_stats['mean']:.4f}%")
                    self.safe_print(f"  涨幅标准差: {resumption_stats['std']:.4f}%")
                    self.safe_print(f"  上涨数量: {resumption_stats['positive_count']} 下跌数量: {resumption_stats['negative_count']}")
                    self.safe_print(f"  涨停数量: {resumption_stats['limit_up_count']} 跌停数量: {resumption_stats['limit_down_count']}")
                    self.safe_print(f"  涨幅超过8%: {resumption_stats['high_change_count']}")
            else:
                resumption_filename = None
                self.safe_print(f"{year}年{month}月: 无复牌日数据")
            
            # 随机采样
            if len(df) <= self.sample_size:
                sampled_df = df
                self.safe_print(f"{year}年{month}月: 总数据{len(df)}行，全部采样")
            else:
                sampled_df = df.sample(n=self.sample_size, random_state=42)
                self.safe_print(f"{year}年{month}月: 总数据{len(df)}行，随机采样{self.sample_size}行")
            
            # 添加采样标识
            sampled_df['SampleYear'] = year
            sampled_df['SampleMonth'] = month
            sampled_df['SampleType'] = 'Random'
            
            # 保存采样数据到文件
            filename = self.output_dir / f"{year}_{month:02d}_sample.csv"
            sampled_df.to_csv(filename, index=False, encoding='utf-8-sig')
            self.safe_print(f"采样数据已保存到: {filename}")
            
            # 打印涨跌幅统计信息
            if stats:
                self.safe_print(f"{year}年{month}月涨跌幅统计:")
                self.safe_print(f"  有效记录: {stats['valid_records']}/{stats['total_records']}")
                self.safe_print(f"  均值: {stats['change_pct_stats']['mean']:.4f}%")
                self.safe_print(f"  中位数: {stats['change_pct_stats']['median']:.4f}%")
                self.safe_print(f"  标准差: {stats['change_pct_stats']['std']:.4f}%")
                self.safe_print(f"  范围: [{stats['change_pct_stats']['min']:.4f}%, {stats['change_pct_stats']['max']:.4f}%]")
                self.safe_print(f"  上涨: {stats['change_pct_stats']['positive_count']:,} 下跌: {stats['change_pct_stats']['negative_count']:,}")
                self.safe_print(f"  涨停: {stats['change_pct_stats']['limit_up_count']:,} 跌停: {stats['change_pct_stats']['limit_down_count']:,}")
                self.safe_print(f"  涨幅超过8%: {stats['change_pct_stats']['high_change_count']:,}")
            
            return {
                'year': year,
                'month': month,
                'total_rows': len(df),
                'sampled_rows': len(sampled_df),
                'filename': str(filename),
                'stats': stats,
                'high_change_filename': str(high_change_filename) if high_change_filename else None,
                'resumption_filename': str(resumption_filename) if resumption_filename else None
            }
            
        except Exception as e:
            self.safe_print(f"采样 {year}年{month}月 数据时出错: {str(e)}")
            return {
                'year': year,
                'month': month,
                'total_rows': 0,
                'sampled_rows': 0,
                'filename': None,
                'error': str(e),
                'stats': {},
                'high_change_filename': None,
                'resumption_filename': None
            }
        finally:
            # 确保数据库连接被关闭
            if 'db' in locals():
                try:
                    db.close()
                except Exception as e:
                    self.safe_print(f"关闭数据库连接时出错: {str(e)}")
    
    def run_sampling(self):
        """运行数据采样"""
        self.safe_print(f"开始股票行情历史表随机采样")
        self.safe_print(f"采样年份: {self.year}")
        self.safe_print(f"每月采样数量: {self.sample_size}")
        self.safe_print(f"线程数: {self.max_workers}")
        
        # 生成月度时间范围
        monthly_ranges = self.generate_monthly_ranges()
        self.safe_print(f"共生成 {len(monthly_ranges)} 个月份的采样任务")
        
        # 使用线程池执行采样
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_month = {
                executor.submit(self.sample_month_data, month_range): month_range 
                for month_range in monthly_ranges
            }
            
            # 收集结果
            for future in as_completed(future_to_month):
                month_range = future_to_month[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.safe_print(f"处理 {month_range['year']}年{month_range['month']}月 时出错: {str(e)}")
        
        # 生成总体报告
        self.generate_overall_report(results)
        
        self.safe_print("数据采样完成")
        return results
    
    def generate_overall_report(self, results):
        """生成总体报告"""
        self.safe_print("生成总体报告...")
        
        # 汇总统计
        total_rows = sum(r['total_rows'] for r in results)
        total_sampled_rows = sum(r['sampled_rows'] for r in results)
        
        # 收集所有采样数据、涨幅超过8%的数据和复牌日数据
        all_sampled_data = []
        all_high_change_data = []
        all_resumption_data = []
        all_stats = []
        
        for result in results:
            if result['filename'] and os.path.exists(result['filename']):
                try:
                    month_data = pd.read_csv(result['filename'])
                    all_sampled_data.append(month_data)
                    
                    if result['stats']:
                        all_stats.append(result['stats'])
                except Exception as e:
                    self.safe_print(f"读取文件 {result['filename']} 时出错: {str(e)}")
            
            if result['high_change_filename'] and os.path.exists(result['high_change_filename']):
                try:
                    high_change_data = pd.read_csv(result['high_change_filename'])
                    all_high_change_data.append(high_change_data)
                except Exception as e:
                    self.safe_print(f"读取文件 {result['high_change_filename']} 时出错: {str(e)}")
            
            if result['resumption_filename'] and os.path.exists(result['resumption_filename']):
                try:
                    resumption_data = pd.read_csv(result['resumption_filename'])
                    all_resumption_data.append(resumption_data)
                except Exception as e:
                    self.safe_print(f"读取文件 {result['resumption_filename']} 时出错: {str(e)}")
        
        # 合并所有采样数据
        if all_sampled_data:
            try:
                combined_samples = pd.concat(all_sampled_data, ignore_index=True)
                combined_filename = self.output_dir / f"{self.year}_all_samples.csv"
                combined_samples.to_csv(combined_filename, index=False, encoding='utf-8-sig')
                self.safe_print(f"所有采样数据已合并保存到: {combined_filename}")
                
                # 计算总体涨跌幅统计
                overall_stats = self.analyze_change_pct(combined_samples)
                if overall_stats:
                    self.safe_print(f"\n=== 总体涨跌幅统计 ===")
                    self.safe_print(f"总记录数: {overall_stats['total_records']:,}")
                    self.safe_print(f"有效记录数: {overall_stats['valid_records']:,}")
                    self.safe_print(f"涨跌幅统计:")
                    self.safe_print(f"  均值: {overall_stats['change_pct_stats']['mean']:.4f}%")
                    self.safe_print(f"  中位数: {overall_stats['change_pct_stats']['median']:.4f}%")
                    self.safe_print(f"  标准差: {overall_stats['change_pct_stats']['std']:.4f}%")
                    self.safe_print(f"  范围: [{overall_stats['change_pct_stats']['min']:.4f}%, {overall_stats['change_pct_stats']['max']:.4f}%]")
                    self.safe_print(f"  四分位数: Q1={overall_stats['change_pct_stats']['q25']:.4f}%, Q3={overall_stats['change_pct_stats']['q75']:.4f}%")
                    self.safe_print(f"  上涨数量: {overall_stats['change_pct_stats']['positive_count']:,}")
                    self.safe_print(f"  下跌数量: {overall_stats['change_pct_stats']['negative_count']:,}")
                    self.safe_print(f"  涨停数量: {overall_stats['change_pct_stats']['limit_up_count']:,}")
                    self.safe_print(f"  跌停数量: {overall_stats['change_pct_stats']['limit_down_count']:,}")
                    self.safe_print(f"  涨幅超过8%: {overall_stats['change_pct_stats']['high_change_count']:,}")
                    
                    # 复牌日统计
                    if 'resumption_stats' in overall_stats:
                        resumption_stats = overall_stats['resumption_stats']
                        self.safe_print(f"\n=== 复牌日涨跌幅统计 ===")
                        self.safe_print(f"复牌日总数: {resumption_stats['count']:,}")
                        self.safe_print(f"平均涨幅: {resumption_stats['mean']:.4f}%")
                        self.safe_print(f"涨幅标准差: {resumption_stats['std']:.4f}%")
                        self.safe_print(f"上涨数量: {resumption_stats['positive_count']:,}")
                        self.safe_print(f"下跌数量: {resumption_stats['negative_count']:,}")
                        self.safe_print(f"涨停数量: {resumption_stats['limit_up_count']:,}")
                        self.safe_print(f"跌停数量: {resumption_stats['limit_down_count']:,}")
                        self.safe_print(f"涨幅超过8%: {resumption_stats['high_change_count']:,}")
                
            except Exception as e:
                self.safe_print(f"合并采样数据时出错: {str(e)}")
        
        # 合并所有涨幅超过8%的数据
        if all_high_change_data:
            try:
                combined_high_change = pd.concat(all_high_change_data, ignore_index=True)
                combined_high_change_filename = self.output_dir / f"{self.year}_all_high_change_8pct.csv"
                combined_high_change.to_csv(combined_high_change_filename, index=False, encoding='utf-8-sig')
                self.safe_print(f"所有涨幅超过8%的数据已合并保存到: {combined_high_change_filename}")
                
                # 统计除权除息情况
                total_high_change = len(combined_high_change)
                adjusting_count = combined_high_change['HasAdjustingFactor'].sum() if 'HasAdjustingFactor' in combined_high_change.columns else 0
                self.safe_print(f"涨幅超过8%数据统计:")
                self.safe_print(f"  总数量: {total_high_change:,}")
                self.safe_print(f"  发生除权除息: {adjusting_count:,}")
                self.safe_print(f"  除权除息比例: {adjusting_count/total_high_change*100:.2f}%" if total_high_change > 0 else "除权除息比例: 0%")
                
            except Exception as e:
                self.safe_print(f"合并涨幅超过8%数据时出错: {str(e)}")
        
        # 合并所有复牌日数据
        if all_resumption_data:
            try:
                combined_resumption = pd.concat(all_resumption_data, ignore_index=True)
                combined_resumption_filename = self.output_dir / f"{self.year}_all_resumption_days.csv"
                combined_resumption.to_csv(combined_resumption_filename, index=False, encoding='utf-8-sig')
                self.safe_print(f"所有复牌日数据已合并保存到: {combined_resumption_filename}")
                
                # 统计复牌日情况
                total_resumption = len(combined_resumption)
                self.safe_print(f"复牌日数据统计:")
                self.safe_print(f"  总数量: {total_resumption:,}")
                
                # 分析复牌原因分布
                if 'SuspendReason' in combined_resumption.columns:
                    reason_counts = combined_resumption['SuspendReason'].value_counts()
                    self.safe_print(f"  主要停牌原因:")
                    for reason, count in reason_counts.head(10).items():
                        self.safe_print(f"    {reason}: {count:,}")
                
            except Exception as e:
                self.safe_print(f"合并复牌日数据时出错: {str(e)}")
        
        # 生成报告文件
        report_content = f"""
股票行情历史表随机采样报告 - 涨跌幅分析与停牌复牌检查
====================================================

采样年份: {self.year}
每月采样数量: {self.sample_size}
线程数: {self.max_workers}
采样时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

总体统计:
- 总数据行数: {total_rows:,}
- 采样数据行数: {total_sampled_rows:,}
- 采样率: {total_sampled_rows/total_rows*100:.2f}% (总行数 > 0时)

涨跌幅分析说明:
- 涨跌幅 = (收盘价 - 前收盘价) / 前收盘价 * 100%
- 涨停: 涨跌幅 >= 9.8%
- 跌停: 涨跌幅 <= -9.8%
- 涨幅超过8%: 涨跌幅 >= 8.0%
- 小幅波动: -1% <= 涨跌幅 <= 1%
- 中等波动: -5% <= 涨跌幅 <= -1% 或 1% <= 涨跌幅 <= 5%
- 大幅波动: -9.8% <= 涨跌幅 <= -5% 或 5% <= 涨跌幅 <= 9.8%

除权除息检查:
- 使用QT_AdjustingFactor表检查涨幅超过8%的数据是否发生除权除息
- 除权除息因子包括精确复权因子、精确复权常数、比例复权因子等

停牌复牌分析:
- 使用LC_SuspendResumption表关联停牌复牌信息
- 分析复牌日的涨跌幅变化特征
- 统计停牌原因分布

月度统计:
"""
        
        for result in sorted(results, key=lambda x: (x['year'], x['month'])):
            if 'error' in result:
                report_content += f"- {result['year']}年{result['month']}月: 错误 - {result['error']}\n"
            else:
                report_content += f"- {result['year']}年{result['month']}月: 总数据{result['total_rows']:,}行, 采样{result['sampled_rows']:,}行"
                if result['stats']:
                    report_content += f", 涨跌幅均值{result['stats']['change_pct_stats']['mean']:.4f}%, 标准差{result['stats']['change_pct_stats']['std']:.4f}%"
                    report_content += f", 涨幅超过8%: {result['stats']['change_pct_stats']['high_change_count']:,}"
                    if 'resumption_stats' in result['stats']:
                        report_content += f", 复牌日: {result['stats']['resumption_stats']['count']}"
                report_content += "\n"
        
        # 保存报告
        try:
            report_file = self.output_dir / f'{self.year}_change_pct_report.txt'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.safe_print(f"涨跌幅分析报告已保存到: {report_file}")
        except Exception as e:
            self.safe_print(f"保存报告文件时出错: {str(e)}")
        
        # 打印关键统计信息
        self.safe_print(f"\n=== 采样完成 ===")
        self.safe_print(f"总数据行数: {total_rows:,}")
        self.safe_print(f"采样数据行数: {total_sampled_rows:,}")
        self.safe_print(f"采样率: {total_sampled_rows/total_rows*100:.2f}%" if total_rows > 0 else "采样率: 0%")
        self.safe_print(f"详细结果保存在: {self.output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='股票行情历史表随机采样 - 涨跌幅分析与停牌复牌检查')
    parser.add_argument('--year', type=int, default=2024, help='采样年份 (默认: 2024)')
    parser.add_argument('--sample-size', type=int, default=1000, help='每月采样数量 (默认: 1000)')
    parser.add_argument('--threads', type=int, default=12, help='线程数 (默认: 12)')
    
    args = parser.parse_args()
    
    # 创建采样器并运行
    sampler = StockQuoteSampler(
        year=args.year,
        sample_size=args.sample_size,
        max_workers=args.threads
    )
    
    results = sampler.run_sampling()


if __name__ == '__main__':
    main() 