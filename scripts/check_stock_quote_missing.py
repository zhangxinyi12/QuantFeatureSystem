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
            'PriceCeiling', 'PriceFloor', 'Ifsuspend', 'SecuCategory',
            'ListedDate', 'ListedState', 'ListedSector'
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
        
        # 根据上市板块设置不同的价格阈值
        self.price_thresholds = {
            1: {'name': '主板', 'min': 0.1, 'max': 5000},      # 主板股票价格通常较低
            6: {'name': '创业板', 'min': 0.01, 'max': 10000},   # 创业板允许高价股
            7: {'name': '科创板', 'min': 0.01, 'max': 10000},   # 科创板允许高价股
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 记录动态阈值配置
        logging.info("动态价格阈值配置:")
        for sector_id, config in self.price_thresholds.items():
            logging.info(f"  板块{sector_id}({config['name']}): 价格范围 {config['min']} - {config['max']}")
    
    def get_sector_name(self, sector_id):
        """获取板块名称"""
        return self.price_thresholds.get(sector_id, {}).get('name', f'未知板块({sector_id})')
    
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
    
    def check_new_stock_issues(self, row, issues, year, month):
        """检查新股上市特殊情况
        
        Args:
            row: 数据行
            issues: 问题列表
            year: 年份
            month: 月份
        """
        try:
            # 检查是否有上市日期
            if pd.isna(row.get('ListedDate')):
                issues.append("新股信息:缺少上市日期")
                logging.warning(f"股票 {row.get('SecuCode', 'Unknown')} 缺少上市日期信息")
                return
            
            listed_date = pd.to_datetime(row['ListedDate'])
            trading_day = pd.to_datetime(row['TradingDay'])
            
            # 计算上市天数
            days_since_listed = (trading_day - listed_date).days
            
            # 检查是否为新股（上市不足30天）
            is_new_stock = days_since_listed <= 30
            
            # 检查涨跌停数据
            price_ceiling = row.get('PriceCeiling')
            price_floor = row.get('PriceFloor')
            
            if is_new_stock:
                # 新股上市前5个交易日不设涨跌幅限制，且数据问题属于正常情况
                if days_since_listed <= 5:
                    # 前5日的数据问题属于正常情况，不标记为问题
                    # 前5日没有涨跌停限制是正常的
                    pass
                else:
                    # 第6个交易日起应有涨跌停限制
                    if price_ceiling is None or price_ceiling == 0:
                        issues.append(f"新股涨跌停:上市{days_since_listed}天应设涨跌停限制(PriceCeiling为空)")
                    if price_floor is None or price_floor == 0:
                        issues.append(f"新股涨跌停:上市{days_since_listed}天应设涨跌停限制(PriceFloor为空)")
                    
                    # 检查涨跌停幅度是否符合规定
                    if price_ceiling is not None and price_ceiling > 0 and row.get('PrevClosePrice', 0) > 0:
                        limit_ratio = (price_ceiling - row['PrevClosePrice']) / row['PrevClosePrice']
                        
                        # 根据板块判断涨跌停幅度
                        listed_sector = row.get('ListedSector')
                        if listed_sector == 1:  # 主板
                            expected_ratio = 0.10  # 10%
                        elif listed_sector in [6, 7]:  # 创业板、科创板
                            expected_ratio = 0.20  # 20%
                        else:
                            expected_ratio = 0.10  # 默认10%
                        
                        if abs(limit_ratio - expected_ratio) > 0.001:  # 允许小误差
                            issues.append(f"新股涨跌停:涨跌停幅度异常(实际{limit_ratio:.3f}, 预期{expected_ratio:.3f})")
            else:
                # 非新股应有正常的涨跌停限制
                if price_ceiling is None or price_ceiling == 0:
                    issues.append("涨跌停:非新股应设涨跌停限制(PriceCeiling为空)")
                if price_floor is None or price_floor == 0:
                    issues.append("涨跌停:非新股应设涨跌停限制(PriceFloor为空)")
            
            # 检查上市状态
            listed_state = row.get('ListedState')
            if listed_state is None:
                issues.append("上市状态:缺少上市状态信息")
            elif listed_state not in [1, 2, 3]:  # 1:正常上市, 2:暂停上市, 3:终止上市
                issues.append(f"上市状态:状态值异常({listed_state})")
            
            # 检查证券类别
            secu_category = row.get('SecuCategory')
            if secu_category is None:
                issues.append("证券类别:缺少证券类别信息")
            elif secu_category != 1:  # 1:股票
                issues.append(f"证券类别:非股票类型({secu_category})")
            
        except Exception as e:
            issues.append(f"新股检查异常: {str(e)}")
    
    def is_new_stock_first_5_days(self, row):
        """判断是否为新股前5个交易日"""
        try:
            if pd.isna(row.get('ListedDate')):
                return False
            
            listed_date = pd.to_datetime(row['ListedDate'])
            trading_day = pd.to_datetime(row['TradingDay'])
            days_since_listed = (trading_day - listed_date).days
            
            # 新股定义为上市不足30天，且前5个交易日
            return days_since_listed <= 30 and days_since_listed <= 5
        except:
            return False
    
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
                c.SecuCode,
                c.SecuMarket,
                c.SecuCategory,
                c.ListedDate,
                c.ListedState,
                c.ListedSector,
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
            WHERE c.SecuCategory = 1 
                AND c.ListedState = 1
                AND c.SecuMarket IN (83, 90)
                AND a.TradingDay BETWEEN '{start}' AND '{end}'
            ORDER BY a.TradingDay, c.SecuCode
            """
            
            # 执行查询
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
            
            # 判断是否为新股前5个交易日
            is_new_stock_first_5 = self.is_new_stock_first_5_days(row)
            
            # 检查每个字段的数据质量
            for field in self.key_fields:
                value = row[field]
                
                # 检查NaN和None
                if pd.isna(value) or value is None:
                    if field in ['PriceCeiling', 'PriceFloor'] and is_new_stock_first_5:
                        continue  # 新股前5日涨跌停限制为空是正常的
                    issues.append(f"{field}:NULL/NaN")
                    continue
                
                # 检查占位符值
                if value in self.placeholder_values:
                    if field in ['PriceCeiling', 'PriceFloor'] and is_new_stock_first_5:
                        continue  # 新股前5日涨跌停限制占位符是正常的
                    issues.append(f"{field}:占位符({value})")
                    continue
                
                # 检查价格字段的0值（停牌时除外）
                if field in self.price_fields and value == 0:
                    if field in ['PriceCeiling', 'PriceFloor'] and is_new_stock_first_5:
                        continue  # 新股前5日涨跌停限制为0是正常的
                    if row['Ifsuspend'] != 1:  # 非停牌状态
                        issues.append(f"{field}:价格为零(非停牌)")
                    else:
                        issues.append(f"{field}:价格为零(停牌)")
                    continue
                
                # 检查成交量字段的0值（停牌时正常）
                if field in self.volume_fields and value == 0:
                    if row['Ifsuspend'] != 1:  # 非停牌状态
                        issues.append(f"{field}:成交量为零(非停牌)")
                    # 停牌时成交量为0是正常的，不标记为问题
                    continue
                
                # 检查价格字段的负值
                if field in self.price_fields and value < 0:
                    issues.append(f"{field}:负价格({value})")
                    continue
                
                # 检查成交量字段的负值
                if field in self.volume_fields and value < 0:
                    issues.append(f"{field}:负成交量({value})")
                    continue
                
                # 检查价格合理性（过高或过低）
                if field in self.price_fields and value > 0:
                    # 获取上市板块的价格阈值
                    listed_sector = row.get('ListedSector')
                    threshold = self.price_thresholds.get(listed_sector, {'name': '默认', 'min': 0.01, 'max': 10000})
                    
                    # 如果不在配置中的板块，记录日志
                    if listed_sector not in self.price_thresholds:
                        logging.debug(f"股票 {row.get('SecuCode', 'Unknown')} 板块 {listed_sector} 不在配置中，使用默认阈值")
                    
                    if value > threshold['max']:  # 价格过高
                        issues.append(f"{field}:价格异常高({value}, 板块{threshold['name']}阈值{threshold['max']})")
                    elif value < threshold['min']:  # 价格过低
                        issues.append(f"{field}:价格异常低({value}, 板块{threshold['name']}阈值{threshold['min']})")
            
            # 检查新股上市特殊情况
            self.check_new_stock_issues(row, issues, year, month)
            
            # 检查价格逻辑关系（新股前5日不检查）
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
                'price_range': [],
                'price_threshold': [],  # 新增：动态阈值问题
                'new_stock': []
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
                    elif '价格异常' in issue and '阈值' in issue:
                        problem_types['price_threshold'].append(row)  # 动态阈值问题
                    elif '价格异常' in issue:
                        problem_types['price_range'].append(row)
                    elif '新股' in issue or '上市' in issue or '涨跌停' in issue:
                        problem_types['new_stock'].append(row)
            
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
- NULL/NaN值检查 (新股前5日涨跌停限制除外)
- 占位符值检查 (-1, -999, -9999, 999, 9999, 99999) (新股前5日涨跌停限制除外)
- 价格字段0值检查 (区分停牌和非停牌，新股前5日涨跌停限制除外)
- 成交量0值检查 (仅非停牌状态，新股前5日除外)
- 负值检查 (新股前5日除外)
- 价格合理性检查 (根据上市板块动态阈值，新股前5日除外)
- 价格逻辑关系检查 (新股前5日除外)
- 新股上市特殊情况检查 (上市日期、涨跌停限制、上市状态)

动态价格阈值配置:
- 板块1(主板): 价格范围 0.1 - 5000
- 板块6(创业板): 价格范围 0.01 - 10000
- 板块7(科创板): 价格范围 0.01 - 10000
- 其他板块: 使用默认阈值 0.01 - 10000

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