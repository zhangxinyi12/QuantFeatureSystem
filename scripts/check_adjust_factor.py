#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票行情历史表完整数据质量检查脚本
支持多线程按月并发查询，检查所有数据质量问题
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connector import JuyuanDB

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/logs/stock_quote_quality_check.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class StockQuoteQualityChecker:
    """股票行情历史表数据质量检查器"""
    
    def __init__(self, start_date='2024-01-01', end_date='2024-12-31', max_workers=12):
        self.start_date = start_date
        self.end_date = end_date
        self.max_workers = max_workers
        self.output_dir = Path('output/processed_data/stock_quote_quality')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 关键字段列表
        self.key_fields = [
            'TradingDay', 'PrevClosePrice', 'OpenPrice', 'HighPrice', 
            'LowPrice', 'ClosePrice', 'TurnoverVolume', 'TurnoverValue',
            'PriceCeiling', 'PriceFloor', 'Ifsuspend', 'AdjustingFactor'
        ]
        
        # 价格字段（不应该为0，除非停牌）
        self.price_fields = [
            'PrevClosePrice', 'OpenPrice', 'HighPrice', 
            'LowPrice', 'ClosePrice', 'PriceCeiling', 'PriceFloor'
        ]
        
        # 成交量字段（可以为0，表示停牌）
        self.volume_fields = ['TurnoverVolume', 'TurnoverValue']
        
        # 异常占位符值
        self.placeholder_values = [-1, -999, -9999, 999, 9999, 99999]
        
        # 线程锁
        self.lock = threading.Lock()
        
    def generate_monthly_ranges(self):
        """生成月度时间范围"""
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        ranges = []
        current = start.replace(day=1)
        
        while current <= end:
            # 计算当月结束日期
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1)
            else:
                next_month = current.replace(month=current.month + 1)
            
            month_end = next_month - timedelta(days=1)
            
            # 确保不超过总体结束日期
            if month_end > end:
                month_end = end
            
            ranges.append({
                'year': current.year,
                'month': current.month,
                'start': current.strftime('%Y-%m-%d'),
                'end': month_end.strftime('%Y-%m-%d')
            })
            
            current = next_month
        
        return ranges
    
    def fill_adjusting_factors(self, df, adj_factors):
        """在Python端补全复权因子（优化版本）
        
        Args:
            df: 基础行情数据DataFrame
            adj_factors: 复权因子数据DataFrame，包含InnerCode, ExDiviDate, RatioAdjustingFactor
            
        Returns:
            添加了AdjustingFactor字段的DataFrame
        """
        if adj_factors.empty:
            # 如果没有复权因子数据，全部设为1.0
            df['AdjustingFactor'] = 1.0
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
        # 这相当于SQL中的窗口函数LAST_VALUE
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
            # 备用方法：使用apply
            return self._fill_adjusting_factors_fallback(df, adj_factors)
    
    def _fill_adjusting_factors_fallback(self, df, adj_factors):
        """备用方法：使用apply进行复权因子补全"""
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
    
    def check_month_data(self, month_range):
        """检查单个月份的数据质量情况"""
        year = month_range['year']
        month = month_range['month']
        start = month_range['start']
        end = month_range['end']
        
        logging.info(f"开始检查 {year}年{month}月 数据质量 ({start} 到 {end})")
        
        try:
            # 创建数据库连接
            db = JuyuanDB()
            
            # 构建查询SQL
            q_key = ['TradingDay', 'PrevClosePrice', 'OpenPrice', 'HighPrice', 
                    'LowPrice', 'ClosePrice', 'TurnoverVolume', 'TurnoverValue']
            l_key = ['PriceCeiling', 'PriceFloor']
            
            q_key_str = ','.join(['a.' + s for s in q_key])
            l_key_str = ','.join(['b.' + s for s in l_key])
            
            # 获取基础行情数据
            sql = f"""
            SELECT 
                a.InnerCode,
                c.SecuCode,
                c.SecuMarket,
                d.Ifsuspend,
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
            
            # 执行查询获取基础数据
            df = db.read_sql(sql)
            
            if df.empty:
                logging.warning(f"{year}年{month}月: 无数据")
                return {
                    'year': year,
                    'month': month,
                    'total_rows': 0,
                    'quality_issues': pd.DataFrame(),
                    'problem_rows': pd.DataFrame(),
                    'summary': {}
                }
            
            # 获取复权因子数据
            adj_factor_sql = f"""
            SELECT 
                InnerCode,
                ExDiviDate,
                RatioAdjustingFactor
            FROM QT_AdjustingFactor
            WHERE ExDiviDate <= '{end}'
            ORDER BY InnerCode, ExDiviDate
            """
            
            adj_factors = db.read_sql(adj_factor_sql)
            
            # 在Python端补全复权因子
            df = self.fill_adjusting_factors(df, adj_factors)
            
            # 检查数据质量
            quality_issues, problem_rows = self.analyze_data_quality(df, year, month)
            
            # 生成统计摘要
            summary = self.generate_summary(df, quality_issues, year, month)
            
            # 保存质量问题数据到文件
            if not quality_issues.empty:
                filename = self.output_dir / f"{year}_{month:02d}_quality_issues.csv"
                quality_issues.to_csv(filename, index=False, encoding='utf-8-sig')
                logging.info(f"质量问题数据已保存到: {filename}")
            
            # 保存原始问题数据到文件
            if not problem_rows.empty:
                filename = self.output_dir / f"{year}_{month:02d}_problem_rows.csv"
                problem_rows.to_csv(filename, index=False, encoding='utf-8-sig')
                logging.info(f"原始问题数据已保存到: {filename}")
            
            logging.info(f"完成 {year}年{month}月 检查，总行数: {len(df)}")
            
            return {
                'year': year,
                'month': month,
                'total_rows': len(df),
                'quality_issues': quality_issues,
                'problem_rows': problem_rows,
                'summary': summary
            }
            
        except Exception as e:
            logging.error(f"检查 {year}年{month}月 数据时出错: {str(e)}")
            return {
                'year': year,
                'month': month,
                'total_rows': 0,
                'quality_issues': pd.DataFrame(),
                'problem_rows': pd.DataFrame(),
                'summary': {},
                'error': str(e)
            }
        finally:
            # 确保数据库连接被关闭
            if 'db' in locals():
                try:
                    db.close()
                except Exception as e:
                    logging.error(f"关闭数据库连接时出错: {str(e)}")
    
    def analyze_data_quality(self, df, year, month):
        """分析数据质量问题"""
        quality_issues = []
        problem_rows = []  # 收集有问题的原始数据行
        
        for idx, row in df.iterrows():
            issues = []
            
            # 检查每个字段的数据质量
            for field in self.key_fields:
                value = row[field]
                
                # 检查NaN和None
                if pd.isna(value) or value is None:
                    issues.append(f"{field}:NULL/NaN")
                    continue
                
                # 检查占位符值
                if value in self.placeholder_values:
                    issues.append(f"{field}:占位符({value})")
                    continue
                
                # 检查价格字段的0值（停牌时除外）
                if field in self.price_fields and value == 0:
                    if row['Ifsuspend'] != 1:  # 非停牌状态
                        issues.append(f"{field}:价格为零(非停牌)")
                    else:
                        issues.append(f"{field}:价格为零(停牌)")
                    continue
                
                # 检查成交量的0值（停牌时正常）
                if field in self.volume_fields and value == 0:
                    if row['Ifsuspend'] != 1:  # 非停牌状态
                        issues.append(f"{field}:成交量为零(非停牌)")
                    # 停牌时成交量为0是正常的，不标记为问题
                    continue
                
                # 检查价格字段的负值
                if field in self.price_fields and value < 0:
                    issues.append(f"{field}:负价格({value})")
                    continue
                
                # 检查成交量的负值
                if field in self.volume_fields and value < 0:
                    issues.append(f"{field}:负成交量({value})")
                    continue
                
                # 检查价格合理性（过高或过低）
                if field in self.price_fields and value > 0:
                    if value > 10000:  # 价格过高
                        issues.append(f"{field}:价格异常高({value})")
                    elif value < 0.01:  # 价格过低
                        issues.append(f"{field}:价格异常低({value})")
                
                # 检查复权因子的合理性
                if field == 'AdjustingFactor':
                    if value <= 0:
                        issues.append(f"{field}:复权因子非正数({value})")
                    elif value > 1000:  # 放宽上限，考虑多次除权的情况
                        issues.append(f"{field}:复权因子异常大({value})")
                    elif value < 0.001:  # 放宽下限，考虑多次送股的情况
                        issues.append(f"{field}:复权因子异常小({value})")
            
            # 检查价格逻辑关系
            if not pd.isna(row['HighPrice']) and not pd.isna(row['LowPrice']) and not pd.isna(row['ClosePrice']):
                if row['HighPrice'] < row['LowPrice']:
                    issues.append("价格逻辑:最高价小于最低价")
                if row['ClosePrice'] > row['HighPrice']:
                    issues.append("价格逻辑:收盘价高于最高价")
                if row['ClosePrice'] < row['LowPrice']:
                    issues.append("价格逻辑:收盘价低于最低价")
            
            if issues:
                # 添加到问题分析列表
                issue_row = {
                    'Year': year,
                    'Month': month,
                    'TradingDay': row['TradingDay'],
                    'SecuCode': row['SecuCode'],
                    'SecuMarket': row['SecuMarket'],
                    'Ifsuspend': row['Ifsuspend'],
                    'Issues': '; '.join(issues),
                    'IssueCount': len(issues)
                }
                
                # 添加所有字段的值
                for field in self.key_fields:
                    issue_row[f'{field}_Value'] = row[field]
                
                quality_issues.append(issue_row)
                
                # 添加到原始问题数据列表
                problem_row = row.copy()
                problem_row['Year'] = year
                problem_row['Month'] = month
                problem_row['Issues'] = '; '.join(issues)
                problem_row['IssueCount'] = len(issues)
                problem_rows.append(problem_row)
        
        return pd.DataFrame(quality_issues), pd.DataFrame(problem_rows)
    
    def generate_summary(self, df, quality_issues, year, month):
        """生成统计摘要"""
        summary = {
            'year': year,
            'month': month,
            'total_rows': len(df),
            'issue_rows': len(quality_issues),
            'issue_rate': len(quality_issues) / len(df) * 100 if len(df) > 0 else 0,
            'field_issue_stats': {},
            'issue_type_stats': {}
        }
        
        # 统计每个字段的问题情况
        for field in self.key_fields:
            null_count = df[field].isna().sum()
            placeholder_count = df[field].isin(self.placeholder_values).sum()
            negative_count = (df[field] < 0).sum() if field in self.price_fields + self.volume_fields else 0
            
            # 对于成交量字段，需要区分停牌和非停牌状态的零值
            if field in self.volume_fields:
                # 停牌时成交量为0是正常的
                zero_suspend = len(df[(df[field] == 0) & (df['Ifsuspend'] == 1)])
                # 非停牌时成交量为0是问题
                zero_normal = len(df[(df[field] == 0) & (df['Ifsuspend'] != 1)])
                zero_count = zero_normal  # 只统计有问题的零值
            else:
                zero_count = (df[field] == 0).sum()
            
            summary['field_issue_stats'][field] = {
                'null_count': null_count,
                'null_rate': null_count / len(df) * 100 if len(df) > 0 else 0,
                'zero_count': zero_count,
                'zero_rate': zero_count / len(df) * 100 if len(df) > 0 else 0,
                'placeholder_count': placeholder_count,
                'placeholder_rate': placeholder_count / len(df) * 100 if len(df) > 0 else 0,
                'negative_count': negative_count,
                'negative_rate': negative_count / len(df) * 100 if len(df) > 0 else 0
            }
        
        # 统计问题类型
        if not quality_issues.empty:
            issue_types = []
            for issues_str in quality_issues['Issues']:
                issue_types.extend([issue.split(':')[0] for issue in issues_str.split('; ')])
            
            from collections import Counter
            issue_counter = Counter(issue_types)
            summary['issue_type_stats'] = dict(issue_counter)
        
        return summary
    
    def run_check(self):
        """运行数据质量检查"""
        logging.info(f"开始股票行情历史表数据质量检查")
        logging.info(f"时间范围: {self.start_date} 到 {self.end_date}")
        logging.info(f"线程数: {self.max_workers}")
        
        # 生成月度时间范围
        monthly_ranges = self.generate_monthly_ranges()
        logging.info(f"共生成 {len(monthly_ranges)} 个月份的检查任务")
        
        # 使用线程池执行检查
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_month = {
                executor.submit(self.check_month_data, month_range): month_range 
                for month_range in monthly_ranges
            }
            
            # 收集结果
            for future in as_completed(future_to_month):
                month_range = future_to_month[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"处理 {month_range['year']}年{month_range['month']}月 时出错: {str(e)}")
        
        # 生成总体报告
        self.generate_overall_report(results)
        
        logging.info("数据质量检查完成")
        return results
    
    def generate_overall_report(self, results):
        """生成总体报告"""
        logging.info("生成总体报告...")
        
        # 汇总统计
        total_rows = sum(r.get('total_rows', 0) for r in results)
        total_issue_rows = sum(r.get('issue_rows', 0) for r in results)
        
        # 收集所有质量问题数据
        all_quality_issues = []
        all_problem_rows = []  # 收集所有原始问题数据
        for result in results:
            if not result.get('quality_issues', pd.DataFrame()).empty:
                all_quality_issues.append(result['quality_issues'])
            if not result.get('problem_rows', pd.DataFrame()).empty:
                all_problem_rows.append(result['problem_rows'])
        
        # 合并所有质量问题数据
        if all_quality_issues:
            combined_issues = pd.concat(all_quality_issues, ignore_index=True)
            combined_issues.to_csv(self.output_dir / 'all_quality_issues.csv', 
                                 index=False, encoding='utf-8-sig')
        
        # 合并所有原始问题数据
        if all_problem_rows:
            combined_problem_rows = pd.concat(all_problem_rows, ignore_index=True)
            combined_problem_rows.to_csv(self.output_dir / 'all_problem_rows.csv', 
                                       index=False, encoding='utf-8-sig')
            print(f"所有原始问题数据已保存到: {self.output_dir / 'all_problem_rows.csv'}")
            print(f"共发现 {len(combined_problem_rows)} 行问题数据")
        else:
            print("未发现任何问题数据")
        
        # 按问题类型分别保存
        if all_problem_rows:
            combined_problem_rows = pd.concat(all_problem_rows, ignore_index=True)
            
            # 按问题类型分组保存
            problem_types = {
                'null_nan': [],
                'placeholder': [],
                'negative': [],
                'zero_price': [],
                'zero_volume': [],
                'price_logic': [],
                'price_range': []
            }
            
            for _, row in combined_problem_rows.iterrows():
                issues = row['Issues'].split('; ')
                for issue in issues:
                    if 'NULL/NaN' in issue:
                        problem_types['null_nan'].append(row)
                    elif '占位符' in issue:
                        problem_types['placeholder'].append(row)
                    elif '负值' in issue or '负价格' in issue or '负成交量' in issue:
                        problem_types['negative'].append(row)
                    elif '价格为零' in issue:
                        problem_types['zero_price'].append(row)
                    elif '成交量为零' in issue:
                        problem_types['zero_volume'].append(row)
                    elif '价格逻辑' in issue:
                        problem_types['price_logic'].append(row)
                    elif '价格异常' in issue:
                        problem_types['price_range'].append(row)
            
            # 保存各类型问题数据
            for problem_type, data in problem_types.items():
                if data:
                    df_type = pd.DataFrame(data)
                    filename = self.output_dir / f'problem_{problem_type}.csv'
                    df_type.to_csv(filename, index=False, encoding='utf-8-sig')
                    print(f"{problem_type} 类型问题数据已保存到: {filename} (共 {len(data)} 行)")
        
        # ========== 详细汇总报告 ==========
        print("\n" + "="*60)
        print("股票行情数据质量检查汇总报告")
        print("="*60)
        print(f"检查时间范围: {self.start_date} 到 {self.end_date}")
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"线程数: {self.max_workers}")
        
        print(f"\n【总体统计】")
        print(f"总数据行数: {total_rows:,}")
        print(f"质量问题行数: {total_issue_rows:,}")
        print(f"问题率: {total_issue_rows/total_rows*100:.2f}%" if total_rows > 0 else "问题率: 0%")
        
        # 问题类型统计
        if all_problem_rows:
            combined_problem_rows = pd.concat(all_problem_rows, ignore_index=True)
            
            print(f"\n【问题类型分布】")
            issue_type_counts = {}
            for _, row in combined_problem_rows.iterrows():
                issues = row['Issues'].split('; ')
                for issue in issues:
                    issue_type = issue.split(':')[0] if ':' in issue else issue
                    issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
            
            for issue_type, count in sorted(issue_type_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"- {issue_type}: {count:,} 次")
        
        # 月度统计
        print(f"\n【月度统计】")
        monthly_stats = {}
        for result in results:
            if 'error' not in result:
                year_month = f"{result.get('year', 'N/A')}年{result.get('month', 'N/A')}月"
                monthly_stats[year_month] = {
                    'total': result.get('total_rows', 0),
                    'issues': result.get('issue_rows', 0),
                    'rate': result.get('issue_rate', 0)
                }
        
        for year_month in sorted(monthly_stats.keys()):
            stats = monthly_stats[year_month]
            print(f"- {year_month}: {stats['total']:,}行, 问题{stats['issues']:,}行 ({stats['rate']:.2f}%)")
        
        # 字段级别统计
        if results and 'field_issue_stats' in results[0].get('summary', {}):
            print(f"\n【字段级别问题统计】")
            for field in self.key_fields:
                total_null = sum(r.get('summary', {}).get('field_issue_stats', {}).get(field, {}).get('null_count', 0) for r in results)
                total_zero = sum(r.get('summary', {}).get('field_issue_stats', {}).get(field, {}).get('zero_count', 0) for r in results)
                total_placeholder = sum(r.get('summary', {}).get('field_issue_stats', {}).get(field, {}).get('placeholder_count', 0) for r in results)
                total_negative = sum(r.get('summary', {}).get('field_issue_stats', {}).get(field, {}).get('negative_count', 0) for r in results)
                
                # 对于成交量字段，需要区分停牌和非停牌状态
                if field in self.volume_fields:
                    # 重新计算成交量字段的零值统计（排除停牌状态）
                    volume_zero_suspend = 0
                    volume_zero_normal = 0
                    for result in results:
                        if not result.get('problem_rows', pd.DataFrame()).empty:
                            problem_df = result['problem_rows']
                            # 停牌时成交量为0是正常的
                            suspend_zero = len(problem_df[(problem_df[field] == 0) & (problem_df['Ifsuspend'] == 1)])
                            # 非停牌时成交量为0是问题
                            normal_zero = len(problem_df[(problem_df[field] == 0) & (problem_df['Ifsuspend'] != 1)])
                            volume_zero_suspend += suspend_zero
                            volume_zero_normal += normal_zero
                    
                    if total_null > 0 or volume_zero_normal > 0 or total_placeholder > 0 or total_negative > 0:
                        print(f"- {field}: NULL={total_null:,}, 零值(非停牌)={volume_zero_normal:,}, 零值(停牌正常)={volume_zero_suspend:,}, 占位符={total_placeholder:,}, 负值={total_negative:,}")
                elif field == 'AdjustingFactor':
                    # 复权因子特殊处理
                    adj_factor_issues = 0
                    for result in results:
                        if not result.get('problem_rows', pd.DataFrame()).empty:
                            problem_df = result['problem_rows']
                            # 统计复权因子相关的问题
                            adj_issues = len(problem_df[problem_df['Issues'].str.contains('AdjustingFactor', na=False)])
                            adj_factor_issues += adj_issues
                    
                    if total_null > 0 or adj_factor_issues > 0 or total_placeholder > 0:
                        print(f"- {field}: NULL={total_null:,}, 问题={adj_factor_issues:,}, 占位符={total_placeholder:,}")
                else:
                    if total_null > 0 or total_zero > 0 or total_placeholder > 0 or total_negative > 0:
                        print(f"- {field}: NULL={total_null:,}, 零值={total_zero:,}, 占位符={total_placeholder:,}, 负值={total_negative:,}")
        
        # 股票问题统计
        if all_problem_rows:
            combined_problem_rows = pd.concat(all_problem_rows, ignore_index=True)
            stock_issue_counts = combined_problem_rows.groupby('SecuCode').size().sort_values(ascending=False)
            
            print(f"\n【问题最多的前10只股票】")
            for stock, count in stock_issue_counts.head(10).items():
                print(f"- {stock}: {count:,} 行问题")
        
        # 交易所统计
        if all_problem_rows:
            market_issue_counts = combined_problem_rows.groupby('SecuMarket').size()
            print(f"\n【交易所问题分布】")
            for market, count in market_issue_counts.items():
                market_name = "深交所" if market == 83 else "上交所" if market == 90 else f"其他({market})"
                print(f"- {market_name}: {count:,} 行问题")
        
        # 问题严重程度分析
        if all_problem_rows:
            combined_problem_rows = pd.concat(all_problem_rows, ignore_index=True)
            issue_count_stats = combined_problem_rows['IssueCount'].describe()
            
            print(f"\n【问题严重程度分析】")
            print(f"- 平均每行问题数: {issue_count_stats['mean']:.2f}")
            print(f"- 最多问题数: {issue_count_stats['max']:.0f}")
            print(f"- 最少问题数: {issue_count_stats['min']:.0f}")
            
            # 多问题行统计
            multi_issue_rows = combined_problem_rows[combined_problem_rows['IssueCount'] > 1]
            print(f"- 多问题行数(>1个问题): {len(multi_issue_rows):,} 行 ({len(multi_issue_rows)/len(combined_problem_rows)*100:.1f}%)")
        
        print(f"\n【建议】")
        if total_issue_rows > 0:
            print("1. 优先处理问题最多的字段")
            print("2. 重点关注问题最多的股票")
            print("3. 检查问题集中的时间段")
            print("4. 对多问题行进行重点审查")
        else:
            print("数据质量良好，未发现明显问题。")
        
        print(f"\n详细结果保存在: {self.output_dir}")
        print("="*60)
        
        # 生成报告文件
        report_content = f"""
股票行情历史表数据质量检查报告
==============================

检查时间范围: {self.start_date} 到 {self.end_date}
检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
线程数: {self.max_workers}

检查项目:
- NULL/NaN值检查
- 占位符值检查 (-1, -999, -9999, 999, 9999, 99999)
- 价格字段0值检查 (区分停牌和非停牌)
- 成交量0值检查 (仅非停牌状态)
- 负值检查
- 价格合理性检查
- 价格逻辑关系检查
- 复权因子合理性检查 (范围: 0.01-100)

总体统计:
- 总数据行数: {total_rows:,}
- 质量问题行数: {total_issue_rows:,}
- 问题率: {total_issue_rows/total_rows*100:.2f}% (总行数 > 0时)

月度统计:
"""
        
        for result in sorted(results, key=lambda x: (x.get('year', 0), x.get('month', 0))):
            if 'error' in result:
                report_content += f"- {result.get('year', 'N/A')}年{result.get('month', 'N/A')}月: 错误 - {result['error']}\n"
            else:
                report_content += f"- {result.get('year', 'N/A')}年{result.get('month', 'N/A')}月: {result.get('total_rows', 0):,}行, 问题{result.get('issue_rows', 0):,}行 ({result.get('issue_rate', 0):.2f}%)\n"
        
        # 字段级别统计
        if results and 'field_issue_stats' in results[0].get('summary', {}):
            report_content += "\n字段级别问题统计:\n"
            for field in self.key_fields:
                total_null = sum(r.get('summary', {}).get('field_issue_stats', {}).get(field, {}).get('null_count', 0) for r in results)
                total_zero = sum(r.get('summary', {}).get('field_issue_stats', {}).get(field, {}).get('zero_count', 0) for r in results)
                total_placeholder = sum(r.get('summary', {}).get('field_issue_stats', {}).get(field, {}).get('placeholder_count', 0) for r in results)
                
                report_content += f"- {field}: NULL={total_null:,}, 零值={total_zero:,}, 占位符={total_placeholder:,}\n"
        
        # 保存报告
        report_file = self.output_dir / 'quality_check_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"总体报告已保存到: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='股票行情历史表数据质量检查')
    parser.add_argument('--start', default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--threads', type=int, default=12, help='线程数 (默认: 12)')
    
    args = parser.parse_args()
    
    # 创建检查器并运行
    checker = StockQuoteQualityChecker(
        start_date=args.start,
        end_date=args.end,
        max_workers=args.threads
    )
    
    results = checker.run_check()


if __name__ == '__main__':
    main() 