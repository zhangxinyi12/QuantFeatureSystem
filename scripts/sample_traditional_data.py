#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票行情历史表随机采样脚本
每月随机采样1000条数据，使用12个线程并发处理
分析换手率和次日涨跌幅之间的关系，生成可视化图表
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    
    def analyze_turnover_next_day_return(self, df):
        """分析换手率和次日涨跌幅的关系"""
        if df.empty:
            return {}
        
        # 按股票代码和日期排序
        df = df.sort_values(['SecuCode', 'TradingDay'])
        
        # 计算次日涨跌幅
        df['NextDayReturn'] = df.groupby('SecuCode')['ChangePCT'].shift(-1)
        
        # 过滤有效数据
        valid_data = df.dropna(subset=['TurnoverRate', 'NextDayReturn'])
        
        if valid_data.empty:
            return {}
        
        # 基本统计
        stats_info = {
            'total_records': len(df),
            'valid_records': len(valid_data),
            'turnover_stats': {
                'mean': valid_data['TurnoverRate'].mean(),
                'median': valid_data['TurnoverRate'].median(),
                'std': valid_data['TurnoverRate'].std(),
                'min': valid_data['TurnoverRate'].min(),
                'max': valid_data['TurnoverRate'].max(),
                'q25': valid_data['TurnoverRate'].quantile(0.25),
                'q75': valid_data['TurnoverRate'].quantile(0.75)
            },
            'next_day_return_stats': {
                'mean': valid_data['NextDayReturn'].mean(),
                'median': valid_data['NextDayReturn'].median(),
                'std': valid_data['NextDayReturn'].std(),
                'min': valid_data['NextDayReturn'].min(),
                'max': valid_data['NextDayReturn'].max()
            }
        }
        
        # 计算相关系数
        correlation = valid_data['TurnoverRate'].corr(valid_data['NextDayReturn'])
        stats_info['correlation'] = correlation
        
        # 分组分析
        # 按换手率分组
        turnover_quantiles = valid_data['TurnoverRate'].quantile([0.2, 0.4, 0.6, 0.8])
        valid_data['TurnoverGroup'] = pd.cut(
            valid_data['TurnoverRate'], 
            bins=[0, turnover_quantiles[0.2], turnover_quantiles[0.4], 
                  turnover_quantiles[0.6], turnover_quantiles[0.8], float('inf')],
            labels=['极低', '较低', '中等', '较高', '极高']
        )
        
        group_stats = valid_data.groupby('TurnoverGroup')['NextDayReturn'].agg([
            'count', 'mean', 'std', 'median'
        ]).round(4)
        
        stats_info['group_analysis'] = group_stats.to_dict()
        
        return stats_info, valid_data
    
    def generate_turnover_analysis_plots(self, valid_data, stats_info, year, month):
        """生成换手率分析图表"""
        if valid_data.empty:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{year}年{month}月 换手率与次日涨跌幅关系分析', fontsize=16)
        
        # 1. 散点图：换手率 vs 次日涨跌幅
        ax1 = axes[0, 0]
        ax1.scatter(valid_data['TurnoverRate'], valid_data['NextDayReturn'], 
                   alpha=0.6, s=20, color='blue')
        ax1.set_xlabel('换手率 (%)')
        ax1.set_ylabel('次日涨跌幅 (%)')
        ax1.set_title(f'散点图 (相关系数: {stats_info["correlation"]:.4f})')
        ax1.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(valid_data['TurnoverRate'], valid_data['NextDayReturn'], 1)
        p = np.poly1d(z)
        ax1.plot(valid_data['TurnoverRate'], p(valid_data['TurnoverRate']), 
                "r--", alpha=0.8, linewidth=2)
        
        # 2. 分组箱线图
        ax2 = axes[0, 1]
        turnover_groups = valid_data.groupby('TurnoverGroup')['NextDayReturn']
        group_data = [group.values for name, group in turnover_groups]
        group_labels = list(turnover_groups.groups.keys())
        
        box_plot = ax2.boxplot(group_data, labels=group_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_xlabel('换手率分组')
        ax2.set_ylabel('次日涨跌幅 (%)')
        ax2.set_title('不同换手率分组的次日涨跌幅分布')
        ax2.grid(True, alpha=0.3)
        
        # 3. 换手率分布直方图
        ax3 = axes[1, 0]
        ax3.hist(valid_data['TurnoverRate'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('换手率 (%)')
        ax3.set_ylabel('频次')
        ax3.set_title('换手率分布')
        ax3.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'均值: {stats_info["turnover_stats"]["mean"]:.2f}%\n'
        stats_text += f'中位数: {stats_info["turnover_stats"]["median"]:.2f}%\n'
        stats_text += f'标准差: {stats_info["turnover_stats"]["std"]:.2f}%'
        ax3.text(0.7, 0.8, stats_text, transform=ax3.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 4. 次日涨跌幅分布直方图
        ax4 = axes[1, 1]
        ax4.hist(valid_data['NextDayReturn'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_xlabel('次日涨跌幅 (%)')
        ax4.set_ylabel('频次')
        ax4.set_title('次日涨跌幅分布')
        ax4.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'均值: {stats_info["next_day_return_stats"]["mean"]:.2f}%\n'
        stats_text += f'中位数: {stats_info["next_day_return_stats"]["median"]:.2f}%\n'
        stats_text += f'标准差: {stats_info["next_day_return_stats"]["std"]:.2f}%'
        ax4.text(0.7, 0.8, stats_text, transform=ax4.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        plot_filename = self.output_dir / f"{year}_{month:02d}_turnover_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.safe_print(f"换手率分析图表已保存到: {plot_filename}")
        
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
            
            # 构建查询SQL - 使用QT_StockPerformance表
            sql = f"""
            SELECT 
                c.SecuCode,
                c.SecuMarket,
                p.TradingDay,
                p.PrevClosePrice,
                p.OpenPrice,
                p.HighPrice,
                p.LowPrice,
                p.ClosePrice,
                p.TurnoverVolume,
                p.TurnoverValue,
                p.ChangePCT,
                p.RangePCT,
                p.TurnoverRate,
                p.TurnoverRateFreeFloat,
                p.AvgPrice
            FROM QT_StockPerformance p
            LEFT JOIN SecuMain c ON p.InnerCode = c.InnerCode
            WHERE (c.SecuCategory = 1 AND c.SecuMarket IN (83, 90) AND c.ListedState = 1)
                AND p.TradingDay BETWEEN '{start}' AND '{end}'
                AND p.TurnoverRate IS NOT NULL
            ORDER BY p.TradingDay, c.SecuCode
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
            
            # 分析换手率和次日涨跌幅关系
            stats_info, valid_data = self.analyze_turnover_next_day_return(df)
            
            # 生成可视化图表
            if not valid_data.empty:
                self.generate_turnover_analysis_plots(valid_data, stats_info, year, month)
            
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
            
            # 保存换手率分析数据
            if not valid_data.empty:
                turnover_filename = self.output_dir / f"{year}_{month:02d}_turnover_analysis.csv"
                valid_data.to_csv(turnover_filename, index=False, encoding='utf-8-sig')
                self.safe_print(f"换手率分析数据已保存到: {turnover_filename}")
            
            # 打印统计信息
            if stats_info:
                self.safe_print(f"{year}年{month}月换手率分析:")
                self.safe_print(f"  有效记录: {stats_info['valid_records']}/{stats_info['total_records']}")
                self.safe_print(f"  换手率均值: {stats_info['turnover_stats']['mean']:.4f}%")
                self.safe_print(f"  次日涨跌幅均值: {stats_info['next_day_return_stats']['mean']:.4f}%")
                self.safe_print(f"  相关系数: {stats_info['correlation']:.4f}")
            
            return {
                'year': year,
                'month': month,
                'total_rows': len(df),
                'sampled_rows': len(sampled_df),
                'filename': str(filename),
                'stats': stats_info
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
        self.safe_print(f"开始股票行情表现数据采样")
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
        all_turnover_data = []
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
            
            # 收集换手率分析数据
            turnover_filename = self.output_dir / f"{result['year']}_{result['month']:02d}_turnover_analysis.csv"
            if turnover_filename.exists():
                try:
                    turnover_data = pd.read_csv(turnover_filename)
                    all_turnover_data.append(turnover_data)
                except Exception as e:
                    self.safe_print(f"读取文件 {turnover_filename} 时出错: {str(e)}")
        
        # 合并所有采样数据
        if all_sampled_data:
            try:
                combined_samples = pd.concat(all_sampled_data, ignore_index=True)
                combined_filename = self.output_dir / f"{self.year}_all_samples.csv"
                combined_samples.to_csv(combined_filename, index=False, encoding='utf-8-sig')
                self.safe_print(f"所有采样数据已合并保存到: {combined_filename}")
            except Exception as e:
                self.safe_print(f"合并采样数据时出错: {str(e)}")
        
        # 合并所有换手率分析数据
        if all_turnover_data:
            try:
                combined_turnover = pd.concat(all_turnover_data, ignore_index=True)
                combined_turnover_filename = self.output_dir / f"{self.year}_all_turnover_analysis.csv"
                combined_turnover.to_csv(combined_turnover_filename, index=False, encoding='utf-8-sig')
                self.safe_print(f"所有换手率分析数据已合并保存到: {combined_turnover_filename}")
                
                # 生成总体换手率分析图表
                overall_stats, overall_valid_data = self.analyze_turnover_next_day_return(combined_turnover)
                if not overall_valid_data.empty:
                    self.generate_turnover_analysis_plots(overall_valid_data, overall_stats, self.year, 0)
                    self.safe_print(f"总体换手率分析图表已保存")
                
            except Exception as e:
                self.safe_print(f"合并换手率分析数据时出错: {str(e)}")
        
        # 生成报告文件
        report_content = f"""
股票行情表现数据采样报告 - 换手率与次日涨跌幅关系分析
==================================================

采样年份: {self.year}
每月采样数量: {self.sample_size}
线程数: {self.max_workers}
采样时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

总体统计:
- 总数据行数: {total_rows:,}
- 采样数据行数: {total_sampled_rows:,}
- 采样率: {total_sampled_rows/total_rows*100:.2f}% (总行数 > 0时)

分析说明:
- 使用QT_StockPerformance表获取股票行情表现数据
- 分析换手率与次日涨跌幅之间的关系
- 生成散点图、箱线图、分布直方图等可视化图表
- 计算相关系数和分组统计

月度统计:
"""
        
        for result in sorted(results, key=lambda x: (x['year'], x['month'])):
            if 'error' in result:
                report_content += f"- {result['year']}年{result['month']}月: 错误 - {result['error']}\n"
            else:
                report_content += f"- {result['year']}年{result['month']}月: 总数据{result['total_rows']:,}行, 采样{result['sampled_rows']:,}行"
                if result['stats']:
                    report_content += f", 换手率均值{result['stats']['turnover_stats']['mean']:.4f}%, 相关系数{result['stats']['correlation']:.4f}"
                report_content += "\n"
        
        # 保存报告
        try:
            report_file = self.output_dir / f'{self.year}_turnover_analysis_report.txt'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.safe_print(f"换手率分析报告已保存到: {report_file}")
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
    
    parser = argparse.ArgumentParser(description='股票行情表现数据采样 - 换手率与次日涨跌幅关系分析')
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