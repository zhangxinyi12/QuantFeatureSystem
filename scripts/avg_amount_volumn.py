#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票行情历史表随机采样脚本
每月随机采样1000条数据，使用12个线程并发处理
计算每只股票的平均成交金额和成交量，以及和low high的比例，得出一个经验比例
计算所有交易日的 (均价 - High)/High（均价高于最高价的比例）和 (Low - 均价)/Low（均价低于最低价的比例）
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
    
    def calculate_average_price_ratios(self, df):
        """计算均价与最高价、最低价的比例"""
        if df.empty:
            return df
        
        # 计算均价 = 成交额/成交量
        df['AveragePrice'] = df['TurnoverValue'] / df['TurnoverVolume']
        
        # 处理除零情况
        df['AveragePrice'] = df['AveragePrice'].replace([np.inf, -np.inf], np.nan)
        
        # 计算比例
        # (均价 - High)/High（均价高于最高价的比例）
        df['HighRatio'] = (df['AveragePrice'] - df['HighPrice']) / df['HighPrice']
        
        # (Low - 均价)/Low（均价低于最低价的比例）
        df['LowRatio'] = (df['LowPrice'] - df['AveragePrice']) / df['LowPrice']
        
        # 处理除零和无效值
        df['HighRatio'] = df['HighRatio'].replace([np.inf, -np.inf], np.nan)
        df['LowRatio'] = df['LowRatio'].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def analyze_price_ratios(self, df):
        """分析价格比例的统计信息"""
        if df.empty:
            return {}
        
        # 过滤有效数据
        valid_data = df.dropna(subset=['AveragePrice', 'HighPrice', 'LowPrice', 'HighRatio', 'LowRatio'])
        
        if valid_data.empty:
            return {}
        
        # 统计信息
        stats = {
            'total_records': len(df),
            'valid_records': len(valid_data),
            'high_ratio_stats': {
                'mean': valid_data['HighRatio'].mean(),
                'median': valid_data['HighRatio'].median(),
                'std': valid_data['HighRatio'].std(),
                'min': valid_data['HighRatio'].min(),
                'max': valid_data['HighRatio'].max(),
                'positive_count': (valid_data['HighRatio'] > 0).sum(),
                'negative_count': (valid_data['HighRatio'] < 0).sum(),
                'zero_count': (valid_data['HighRatio'] == 0).sum()
            },
            'low_ratio_stats': {
                'mean': valid_data['LowRatio'].mean(),
                'median': valid_data['LowRatio'].median(),
                'std': valid_data['LowRatio'].std(),
                'min': valid_data['LowRatio'].min(),
                'max': valid_data['LowRatio'].max(),
                'positive_count': (valid_data['LowRatio'] > 0).sum(),
                'negative_count': (valid_data['LowRatio'] < 0).sum(),
                'zero_count': (valid_data['LowRatio'] == 0).sum()
            }
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
                    'stats': {}
                }
            
            # 计算均价和比例
            df = self.calculate_average_price_ratios(df)
            
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
            
            # 分析价格比例统计
            stats = self.analyze_price_ratios(sampled_df)
            
            # 保存采样数据到文件
            filename = self.output_dir / f"{year}_{month:02d}_sample.csv"
            sampled_df.to_csv(filename, index=False, encoding='utf-8-sig')
            self.safe_print(f"采样数据已保存到: {filename}")
            
            # 打印统计信息
            if stats:
                self.safe_print(f"{year}年{month}月价格比例统计:")
                self.safe_print(f"  有效记录: {stats['valid_records']}/{stats['total_records']}")
                self.safe_print(f"  高价比例均值: {stats['high_ratio_stats']['mean']:.4f}")
                self.safe_print(f"  低价比例均值: {stats['low_ratio_stats']['mean']:.4f}")
            
            return {
                'year': year,
                'month': month,
                'total_rows': len(df),
                'sampled_rows': len(sampled_df),
                'filename': str(filename),
                'stats': stats
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
                'stats': {}
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
        
        # 收集所有采样数据
        all_sampled_data = []
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
        
        # 合并所有采样数据
        if all_sampled_data:
            try:
                combined_samples = pd.concat(all_sampled_data, ignore_index=True)
                combined_filename = self.output_dir / f"{self.year}_all_samples.csv"
                combined_samples.to_csv(combined_filename, index=False, encoding='utf-8-sig')
                self.safe_print(f"所有采样数据已合并保存到: {combined_filename}")
                
                # 计算总体价格比例统计
                overall_stats = self.analyze_price_ratios(combined_samples)
                if overall_stats:
                    self.safe_print(f"\n=== 总体价格比例统计 ===")
                    self.safe_print(f"总记录数: {overall_stats['total_records']:,}")
                    self.safe_print(f"有效记录数: {overall_stats['valid_records']:,}")
                    self.safe_print(f"高价比例统计:")
                    self.safe_print(f"  均值: {overall_stats['high_ratio_stats']['mean']:.4f}")
                    self.safe_print(f"  中位数: {overall_stats['high_ratio_stats']['median']:.4f}")
                    self.safe_print(f"  标准差: {overall_stats['high_ratio_stats']['std']:.4f}")
                    self.safe_print(f"  范围: [{overall_stats['high_ratio_stats']['min']:.4f}, {overall_stats['high_ratio_stats']['max']:.4f}]")
                    self.safe_print(f"  正值数量: {overall_stats['high_ratio_stats']['positive_count']:,}")
                    self.safe_print(f"  负值数量: {overall_stats['high_ratio_stats']['negative_count']:,}")
                    
                    self.safe_print(f"低价比例统计:")
                    self.safe_print(f"  均值: {overall_stats['low_ratio_stats']['mean']:.4f}")
                    self.safe_print(f"  中位数: {overall_stats['low_ratio_stats']['median']:.4f}")
                    self.safe_print(f"  标准差: {overall_stats['low_ratio_stats']['std']:.4f}")
                    self.safe_print(f"  范围: [{overall_stats['low_ratio_stats']['min']:.4f}, {overall_stats['low_ratio_stats']['max']:.4f}]")
                    self.safe_print(f"  正值数量: {overall_stats['low_ratio_stats']['positive_count']:,}")
                    self.safe_print(f"  负值数量: {overall_stats['low_ratio_stats']['negative_count']:,}")
                
            except Exception as e:
                self.safe_print(f"合并采样数据时出错: {str(e)}")
        
        # 生成报告文件
        report_content = f"""
股票行情历史表随机采样报告
==========================

采样年份: {self.year}
每月采样数量: {self.sample_size}
线程数: {self.max_workers}
采样时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

总体统计:
- 总数据行数: {total_rows:,}
- 采样数据行数: {total_sampled_rows:,}
- 采样率: {total_sampled_rows/total_rows*100:.2f}% (总行数 > 0时)

价格比例分析:
- 均价计算公式: 成交额/成交量
- 高价比例: (均价 - High)/High
- 低价比例: (Low - 均价)/Low

月度统计:
"""
        
        for result in sorted(results, key=lambda x: (x['year'], x['month'])):
            if 'error' in result:
                report_content += f"- {result['year']}年{result['month']}月: 错误 - {result['error']}\n"
            else:
                report_content += f"- {result['year']}年{result['month']}月: 总数据{result['total_rows']:,}行, 采样{result['sampled_rows']:,}行"
                if result['stats']:
                    report_content += f", 高价比例均值{result['stats']['high_ratio_stats']['mean']:.4f}, 低价比例均值{result['stats']['low_ratio_stats']['mean']:.4f}"
                report_content += "\n"
        
        # 保存报告
        try:
            report_file = self.output_dir / f'{self.year}_sampling_report.txt'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.safe_print(f"采样报告已保存到: {report_file}")
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
    
    parser = argparse.ArgumentParser(description='股票行情历史表随机采样')
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