#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析开盘价和前一日收盘价比率的脚本
用于处理开盘价缺失时的填充策略
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
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
        logging.FileHandler('output/logs/open_close_ratio_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class OpenCloseRatioAnalyzer:
    """开盘价和前一日收盘价比率分析器"""
    
    def __init__(self, start_date='2024-01-01', end_date='2024-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path('output/processed_data/open_close_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def fetch_data(self):
        """获取股票数据"""
        logging.info(f"开始获取数据: {self.start_date} 到 {self.end_date}")
        
        try:
            db = JuyuanDB()
            
            # 查询股票数据，包含前一日收盘价
            sql = f"""
            SELECT 
                c.SecuCode,
                c.SecuMarket,
                c.ListedSector,
                a.TradingDay,
                a.OpenPrice,
                a.ClosePrice,
                LAG(a.ClosePrice) OVER (PARTITION BY a.InnerCode ORDER BY a.TradingDay) as PrevClosePrice,
                a.Ifsuspend
            FROM QT_DailyQuote a
            LEFT JOIN SecuMain c ON a.InnerCode = c.InnerCode
            WHERE c.SecuCategory = 1 
                AND c.ListedState = 1
                AND c.SecuMarket IN (83, 90)
                AND a.TradingDay BETWEEN '{self.start_date}' AND '{self.end_date}'
                AND a.OpenPrice IS NOT NULL
                AND a.ClosePrice IS NOT NULL
            ORDER BY a.TradingDay, c.SecuCode
            """
            
            df = db.read_sql(sql)
            db.close()
            
            logging.info(f"获取数据完成，共 {len(df)} 行")
            return df
            
        except Exception as e:
            logging.error(f"获取数据失败: {str(e)}")
            return pd.DataFrame()
    
    def fetch_data_with_sampling(self):
        """获取股票数据（使用随机采样）"""
        logging.info(f"开始获取数据（随机采样）: {self.start_date} 到 {self.end_date}")
        
        try:
            db = JuyuanDB()
            
            # 生成月度时间范围
            monthly_ranges = self.generate_monthly_ranges()
            all_sampled_data = []
            
            for month_range in monthly_ranges:
                year = month_range['year']
                month = month_range['month']
                start = month_range['start']
                end = month_range['end']
                
                logging.info(f"采样 {year}年{month}月 数据 ({start} 到 {end})")
                
                # 查询单月数据
                sql = f"""
                SELECT 
                    c.SecuCode,
                    c.SecuMarket,
                    c.ListedSector,
                    a.TradingDay,
                    a.OpenPrice,
                    a.ClosePrice,
                    LAG(a.ClosePrice) OVER (PARTITION BY a.InnerCode ORDER BY a.TradingDay) as PrevClosePrice,
                    a.Ifsuspend
                FROM QT_DailyQuote a
                LEFT JOIN SecuMain c ON a.InnerCode = c.InnerCode
                WHERE c.SecuCategory = 1 
                    AND c.ListedState = 1
                    AND c.SecuMarket IN (83, 90)
                    AND a.TradingDay BETWEEN '{start}' AND '{end}'
                    AND a.OpenPrice IS NOT NULL
                    AND a.ClosePrice IS NOT NULL
                ORDER BY a.TradingDay, c.SecuCode
                """
                
                month_df = db.read_sql(sql)
                
                if not month_df.empty:
                    # 随机采样1000条数据
                    sample_size = min(1000, len(month_df))
                    if len(month_df) > sample_size:
                        sampled_df = month_df.sample(n=sample_size, random_state=42)
                        logging.info(f"{year}年{month}月: 总数据{len(month_df)}行，随机采样{sample_size}行")
                    else:
                        sampled_df = month_df
                        logging.info(f"{year}年{month}月: 总数据{len(month_df)}行，全部采样")
                    
                    # 添加采样标识
                    sampled_df['SampleYear'] = year
                    sampled_df['SampleMonth'] = month
                    sampled_df['SampleType'] = 'Random'
                    
                    all_sampled_data.append(sampled_df)
                else:
                    logging.warning(f"{year}年{month}月: 无数据")
            
            db.close()
            
            if all_sampled_data:
                # 合并所有采样数据
                df = pd.concat(all_sampled_data, ignore_index=True)
                logging.info(f"采样数据获取完成，共 {len(df)} 行")
                
                # 保存采样数据
                sample_file = self.output_dir / 'sampled_data.csv'
                df.to_csv(sample_file, index=False, encoding='utf-8-sig')
                logging.info(f"采样数据已保存到: {sample_file}")
                
                return df
            else:
                logging.error("所有月份都没有数据")
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"获取采样数据失败: {str(e)}")
            return pd.DataFrame()
    
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
    
    def calculate_ratios(self, df):
        """计算开盘价和前一日收盘价的比率"""
        logging.info("开始计算比率...")
        
        # 过滤有效数据
        valid_df = df[
            (df['OpenPrice'] > 0) & 
            (df['PrevClosePrice'] > 0) & 
            (df['Ifsuspend'] != 1)  # 排除停牌数据
        ].copy()
        
        # 计算比率
        valid_df['OpenCloseRatio'] = valid_df['OpenPrice'] / valid_df['PrevClosePrice']
        valid_df['OpenCloseRatioPct'] = (valid_df['OpenCloseRatio'] - 1) * 100  # 百分比
        
        # 计算对数收益率
        valid_df['LogReturn'] = np.log(valid_df['OpenCloseRatio'])
        
        logging.info(f"有效数据行数: {len(valid_df)}")
        return valid_df
    
    def analyze_distribution(self, df):
        """分析比率分布"""
        logging.info("分析比率分布...")
        
        # 基本统计
        stats = {
            '样本数': len(df),
            '平均比率': df['OpenCloseRatio'].mean(),
            '中位数比率': df['OpenCloseRatio'].median(),
            '标准差': df['OpenCloseRatio'].std(),
            '最小值': df['OpenCloseRatio'].min(),
            '最大值': df['OpenCloseRatio'].max(),
            '25%分位数': df['OpenCloseRatio'].quantile(0.25),
            '75%分位数': df['OpenCloseRatio'].quantile(0.75),
            '95%分位数': df['OpenCloseRatio'].quantile(0.95),
            '99%分位数': df['OpenCloseRatio'].quantile(0.99),
        }
        
        # 百分比统计
        pct_stats = {
            '平均涨跌幅(%)': df['OpenCloseRatioPct'].mean(),
            '中位数涨跌幅(%)': df['OpenCloseRatioPct'].median(),
            '涨跌幅标准差(%)': df['OpenCloseRatioPct'].std(),
            '最大涨幅(%)': df['OpenCloseRatioPct'].max(),
            '最大跌幅(%)': df['OpenCloseRatioPct'].min(),
        }
        
        # 合并统计
        all_stats = {**stats, **pct_stats}
        
        # 保存统计结果
        stats_df = pd.DataFrame(list(all_stats.items()), columns=['指标', '值'])
        stats_file = self.output_dir / 'open_close_ratio_stats.csv'
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        logging.info(f"统计结果已保存到: {stats_file}")
        
        return all_stats
    
    def analyze_by_sector(self, df):
        """按板块分析比率"""
        logging.info("按板块分析比率...")
        
        sector_stats = []
        
        for sector_id in df['ListedSector'].unique():
            if pd.isna(sector_id):
                continue
                
            sector_data = df[df['ListedSector'] == sector_id]
            
            if len(sector_data) == 0:
                continue
            
            sector_name = self.get_sector_name(sector_id)
            
            stats = {
                '板块ID': sector_id,
                '板块名称': sector_name,
                '样本数': len(sector_data),
                '平均比率': sector_data['OpenCloseRatio'].mean(),
                '中位数比率': sector_data['OpenCloseRatio'].median(),
                '标准差': sector_data['OpenCloseRatio'].std(),
                '平均涨跌幅(%)': sector_data['OpenCloseRatioPct'].mean(),
                '中位数涨跌幅(%)': sector_data['OpenCloseRatioPct'].median(),
            }
            
            sector_stats.append(stats)
        
        sector_df = pd.DataFrame(sector_stats)
        sector_file = self.output_dir / 'sector_ratio_stats.csv'
        sector_df.to_csv(sector_file, index=False, encoding='utf-8-sig')
        logging.info(f"板块统计已保存到: {sector_file}")
        
        return sector_df
    
    def analyze_extreme_cases(self, df):
        """分析极端情况"""
        logging.info("分析极端情况...")
        
        # 找出极端比率（超过3个标准差）
        mean_ratio = df['OpenCloseRatio'].mean()
        std_ratio = df['OpenCloseRatio'].std()
        threshold = 3 * std_ratio
        
        extreme_cases = df[
            (df['OpenCloseRatio'] > mean_ratio + threshold) |
            (df['OpenCloseRatio'] < mean_ratio - threshold)
        ].copy()
        
        extreme_cases = extreme_cases.sort_values('OpenCloseRatio', ascending=False)
        
        # 保存极端情况
        extreme_file = self.output_dir / 'extreme_open_close_ratios.csv'
        extreme_cases.to_csv(extreme_file, index=False, encoding='utf-8-sig')
        logging.info(f"极端情况已保存到: {extreme_file}，共 {len(extreme_cases)} 条")
        
        return extreme_cases
    
    def generate_fill_strategies(self, stats):
        """生成开盘价缺失时的填充策略"""
        logging.info("生成填充策略...")
        
        strategies = {
            '策略1_简单前收': {
                '描述': '直接用前一日收盘价填充',
                '公式': 'OpenPrice = PrevClosePrice',
                '适用场景': '一般情况，简单快速',
                '优点': '简单、快速、无偏差',
                '缺点': '忽略了开盘价通常与收盘价有差异的事实'
            },
            '策略2_平均比率': {
                '描述': '使用历史平均比率调整',
                '公式': f'OpenPrice = PrevClosePrice * {stats["平均比率"]:.4f}',
                '适用场景': '需要反映历史平均开盘价特征',
                '优点': '考虑了历史平均开盘价特征',
                '缺点': '可能不够精确，忽略了市场状态'
            },
            '策略3_中位数比率': {
                '描述': '使用历史中位数比率调整',
                '公式': f'OpenPrice = PrevClosePrice * {stats["中位数比率"]:.4f}',
                '适用场景': '需要稳健的填充策略',
                '优点': '对异常值不敏感，更稳健',
                '缺点': '可能不够精确'
            },
            '策略4_板块特定比率': {
                '描述': '使用板块特定的平均比率',
                '公式': 'OpenPrice = PrevClosePrice * SectorAvgRatio',
                '适用场景': '不同板块开盘价特征差异较大时',
                '优点': '考虑了板块特征',
                '缺点': '需要板块数据，计算复杂'
            },
            '策略5_随机波动': {
                '描述': '在平均比率基础上添加随机波动',
                '公式': f'OpenPrice = PrevClosePrice * ({stats["平均比率"]:.4f} + random_noise)',
                '适用场景': '需要模拟真实开盘价的随机性',
                '优点': '更接近真实开盘价的随机特征',
                '缺点': '引入了随机性，结果不稳定'
            },
            '策略6_涨跌停限制': {
                '描述': '考虑涨跌停限制的填充',
                '公式': 'OpenPrice = min(max(PrevClosePrice * ratio, PriceFloor), PriceCeiling)',
                '适用场景': '有涨跌停限制的市场',
                '优点': '符合市场规则',
                '缺点': '需要涨跌停数据'
            }
        }
        
        # 保存策略
        strategies_file = self.output_dir / 'fill_strategies.md'
        with open(strategies_file, 'w', encoding='utf-8') as f:
            f.write("# 开盘价缺失填充策略\n\n")
            f.write(f"基于 {self.start_date} 到 {self.end_date} 的数据分析\n\n")
            f.write("## 关键统计指标\n\n")
            f.write(f"- 平均开盘价/前收价比率: {stats['平均比率']:.4f}\n")
            f.write(f"- 中位数比率: {stats['中位数比率']:.4f}\n")
            f.write(f"- 标准差: {stats['标准差']:.4f}\n")
            f.write(f"- 平均涨跌幅: {stats['平均涨跌幅(%)']:.2f}%\n\n")
            
            f.write("## 推荐策略\n\n")
            f.write("### 1. 快速填充（推荐用于大量数据）\n")
            f.write(f"- 使用前一日收盘价 * {stats['平均比率']:.4f}\n")
            f.write(f"- 简单有效，偏差较小\n\n")
            
            f.write("### 2. 精确填充（推荐用于重要数据）\n")
            f.write("- 使用板块特定的平均比率\n")
            f.write("- 考虑涨跌停限制\n")
            f.write("- 添加适当的随机波动\n\n")
            
            f.write("## 详细策略说明\n\n")
            
            for strategy_name, strategy in strategies.items():
                f.write(f"### {strategy_name}\n")
                f.write(f"- **描述**: {strategy['描述']}\n")
                f.write(f"- **公式**: {strategy['公式']}\n")
                f.write(f"- **适用场景**: {strategy['适用场景']}\n")
                f.write(f"- **优点**: {strategy['优点']}\n")
                f.write(f"- **缺点**: {strategy['缺点']}\n\n")
        
        logging.info(f"填充策略已保存到: {strategies_file}")
        return strategies
    
    def create_visualizations(self, df):
        """创建可视化图表"""
        logging.info("创建可视化图表...")
        
        # 设置图表样式
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('开盘价与前一日收盘价比率分析', fontsize=16, fontweight='bold')
        
        # 1. 比率分布直方图
        axes[0, 0].hist(df['OpenCloseRatio'], bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(df['OpenCloseRatio'].mean(), color='red', linestyle='--', label=f'均值: {df["OpenCloseRatio"].mean():.4f}')
        axes[0, 0].axvline(df['OpenCloseRatio'].median(), color='orange', linestyle='--', label=f'中位数: {df["OpenCloseRatio"].median():.4f}')
        axes[0, 0].set_xlabel('开盘价/前收价比率')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('比率分布直方图')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 涨跌幅分布
        axes[0, 1].hist(df['OpenCloseRatioPct'], bins=100, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(df['OpenCloseRatioPct'].mean(), color='red', linestyle='--', label=f'均值: {df["OpenCloseRatioPct"].mean():.2f}%')
        axes[0, 1].axvline(df['OpenCloseRatioPct'].median(), color='orange', linestyle='--', label=f'中位数: {df["OpenCloseRatioPct"].median():.2f}%')
        axes[0, 1].set_xlabel('开盘涨跌幅 (%)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('开盘涨跌幅分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 对数收益率分布
        axes[1, 0].hist(df['LogReturn'], bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].axvline(df['LogReturn'].mean(), color='red', linestyle='--', label=f'均值: {df["LogReturn"].mean():.4f}')
        axes[1, 0].set_xlabel('对数收益率')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('对数收益率分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 按板块的箱线图
        sector_data = []
        sector_labels = []
        
        for sector_id in sorted(df['ListedSector'].unique()):
            if pd.isna(sector_id):
                continue
            sector_name = self.get_sector_name(sector_id)
            sector_ratios = df[df['ListedSector'] == sector_id]['OpenCloseRatio']
            if len(sector_ratios) > 0:
                sector_data.append(sector_ratios.values)
                sector_labels.append(f'{sector_id}({sector_name})')
        
        if sector_data:
            axes[1, 1].boxplot(sector_data, labels=sector_labels)
            axes[1, 1].set_ylabel('开盘价/前收价比率')
            axes[1, 1].set_title('各板块比率分布')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.output_dir / 'open_close_ratio_analysis.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        logging.info(f"可视化图表已保存到: {chart_file}")
        
        plt.show()
    
    def get_sector_name(self, sector_id):
        """获取板块名称"""
        sector_names = {
            1: '主板',
            6: '创业板',
            7: '科创板'
        }
        return sector_names.get(sector_id, f'未知板块({sector_id})')
    
    def run_analysis(self, use_sampling=True):
        """运行完整分析"""
        logging.info("开始开盘价与前一日收盘价比率分析")
        
        # 1. 获取数据
        if use_sampling:
            logging.info("使用随机采样模式")
            df = self.fetch_data_with_sampling()
        else:
            logging.info("使用全量数据模式")
            df = self.fetch_data()
            
        if df.empty:
            logging.error("无法获取数据，分析终止")
            return
        
        # 2. 计算比率
        df_with_ratios = self.calculate_ratios(df)
        if df_with_ratios.empty:
            logging.error("无法计算比率，分析终止")
            return
        
        # 3. 分析分布
        stats = self.analyze_distribution(df_with_ratios)
        
        # 4. 按板块分析
        sector_stats = self.analyze_by_sector(df_with_ratios)
        
        # 5. 分析极端情况
        extreme_cases = self.analyze_extreme_cases(df_with_ratios)
        
        # 6. 分析波动率和噪声填充范围
        volatility_stats, error_distribution = self.analyze_volatility_and_noise(df_with_ratios)
        
        # 7. 生成填充策略
        strategies = self.generate_fill_strategies(stats)
        
        # 8. 生成噪声填充策略
        noise_strategies = self.generate_noise_fill_strategies(volatility_stats, error_distribution)
        
        # 9. 创建可视化
        self.create_visualizations(df_with_ratios)
        
        # 10. 创建波动率可视化
        self.create_volatility_visualizations(df_with_ratios, error_distribution)
        
        # 11. 分析采样统计
        sampling_stats, overall_sampling_stats = self.analyze_sampling_statistics(df)
        
        # 12. 输出总结
        self.print_summary(stats, sector_stats, extreme_cases, volatility_stats, error_distribution, sampling_stats, overall_sampling_stats)
        
        logging.info("分析完成")
    
    def print_summary(self, stats, sector_stats, extreme_cases, volatility_stats, error_distribution, sampling_stats, overall_sampling_stats):
        """打印分析总结"""
        print("\n" + "="*60)
        print("开盘价与前一日收盘价比率分析总结")
        print("="*60)
        
        print(f"\n📊 基本统计:")
        print(f"  样本数: {stats['样本数']:,}")
        print(f"  平均比率: {stats['平均比率']:.4f}")
        print(f"  中位数比率: {stats['中位数比率']:.4f}")
        print(f"  标准差: {stats['标准差']:.4f}")
        print(f"  平均涨跌幅: {stats['平均涨跌幅(%)']:.2f}%")
        
        print(f"\n📈 分位数统计:")
        print(f"  25%分位数: {stats['25%分位数']:.4f}")
        print(f"  75%分位数: {stats['75%分位数']:.4f}")
        print(f"  95%分位数: {stats['95%分位数']:.4f}")
        print(f"  99%分位数: {stats['99%分位数']:.4f}")
        
        print(f"\n🏢 板块分析:")
        for _, row in sector_stats.iterrows():
            print(f"  {row['板块名称']}: 平均比率={row['平均比率']:.4f}, 样本数={row['样本数']:,}")
        
        print(f"\n⚠️  极端情况:")
        print(f"  极端比率数量: {len(extreme_cases):,}")
        print(f"  极端比率占比: {len(extreme_cases)/stats['样本数']*100:.2f}%")
        
        print(f"\n📊 波动率分析:")
        print(f"  波动率(标准差): {volatility_stats['波动率(标准差)']:.6f}")
        print(f"  波动率百分比: {volatility_stats['波动率(%)']:.4f}%")
        print(f"  ±0.1%噪声覆盖率: {volatility_stats['±0.1%覆盖率(%)']:.2f}%")
        print(f"  ±1.2%误差覆盖率: {volatility_stats['±1.2%覆盖率(%)']:.2f}%")
        
        print(f"\n📈 误差分布:")
        for error_info in error_distribution:
            print(f"  {error_info['误差范围']}: {error_info['数据量']:,} 条 ({error_info['覆盖率(%)']:.2f}%)")
        
        print(f"\n📊 采样统计:")
        for _, row in sampling_stats.iterrows():
            print(f"  采样月份: {row['年月']}")
            print(f"  样本数: {row['样本数']:,}")
            print(f"  平均比率: {row['平均比率']:.4f}")
            print(f"  比率标准差: {row['比率标准差']:.4f}")
            print(f"  最小比率: {row['最小比率']:.4f}")
            print(f"  最大比率: {row['最大比率']:.4f}")
            print(f"  平均涨跌幅: {row['平均涨跌幅']:.2f}%")
            print(f"  涨跌幅标准差: {row['涨跌幅标准差']:.2f}%")
            print(f"  最小涨跌幅: {row['最小涨跌幅']:.2f}%")
            print(f"  最大涨跌幅: {row['最大涨跌幅']:.2f}%")
        
        print(f"\n📊 总体采样统计:")
        for key, value in overall_sampling_stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n💡 推荐填充策略:")
        print(f"  1. 快速填充: 前收价 * {stats['平均比率']:.4f}")
        print(f"  2. 稳健填充: 前收价 * {stats['中位数比率']:.4f}")
        print(f"  3. 精确填充: 使用板块特定比率")
        
        print(f"\n🎯 噪声填充策略:")
        print(f"  1. 保守噪声(±0.1%): 覆盖率 {volatility_stats['±0.1%覆盖率(%)']:.2f}%")
        print(f"  2. 标准噪声(±{volatility_stats['波动率(%)']:.4f}%): 覆盖率约68%")
        print(f"  3. 安全噪声(±1.2%): 覆盖率 {volatility_stats['±1.2%覆盖率(%)']:.2f}%")
        
        print(f"\n📁 结果文件:")
        print(f"  统计结果: {self.output_dir / 'open_close_ratio_stats.csv'}")
        print(f"  板块统计: {self.output_dir / 'sector_ratio_stats.csv'}")
        print(f"  极端情况: {self.output_dir / 'extreme_open_close_ratios.csv'}")
        print(f"  波动率分析: {self.output_dir / 'volatility_analysis.csv'}")
        print(f"  误差分布: {self.output_dir / 'error_distribution.csv'}")
        print(f"  填充策略: {self.output_dir / 'fill_strategies.md'}")
        print(f"  噪声策略: {self.output_dir / 'noise_fill_strategies.md'}")
        print(f"  可视化图: {self.output_dir / 'open_close_ratio_analysis.png'}")
        print(f"  波动率图: {self.output_dir / 'volatility_analysis.png'}")
        print(f"  采样统计: {self.output_dir / 'sampling_statistics.csv'}")
        print(f"  总体采样统计: {self.output_dir / 'overall_sampling_stats.csv'}")
        
        print("="*60)
    
    def analyze_volatility_and_noise(self, df):
        """分析波动率和噪声填充范围"""
        logging.info("分析波动率和噪声填充范围...")
        
        # 计算波动率（标准差）
        volatility = df['OpenCloseRatio'].std()
        volatility_pct = volatility * 100  # 转换为百分比
        
        # 计算±0.1%噪声范围
        noise_range = 0.001  # ±0.1%
        noise_lower = 1.0 - noise_range
        noise_upper = 1.0 + noise_range
        
        # 计算在噪声范围内的数据比例
        within_noise_range = df[
            (df['OpenCloseRatio'] >= noise_lower) & 
            (df['OpenCloseRatio'] <= noise_upper)
        ]
        noise_coverage = len(within_noise_range) / len(df) * 100
        
        # 计算波动率误差分析
        volatility_error_threshold = 0.012  # 1.2%
        within_volatility_error = df[
            (df['OpenCloseRatio'] >= 1.0 - volatility_error_threshold) & 
            (df['OpenCloseRatio'] <= 1.0 + volatility_error_threshold)
        ]
        volatility_error_coverage = len(within_volatility_error) / len(df) * 100
        
        # 计算不同误差范围的数据分布
        error_ranges = [
            (0.001, '±0.1%'),
            (0.002, '±0.2%'),
            (0.005, '±0.5%'),
            (0.01, '±1.0%'),
            (0.012, '±1.2%'),
            (0.02, '±2.0%'),
            (0.05, '±5.0%')
        ]
        
        error_distribution = []
        for error_range, label in error_ranges:
            within_range = df[
                (df['OpenCloseRatio'] >= 1.0 - error_range) & 
                (df['OpenCloseRatio'] <= 1.0 + error_range)
            ]
            coverage = len(within_range) / len(df) * 100
            error_distribution.append({
                '误差范围': label,
                '数值范围': error_range,
                '数据量': len(within_range),
                '覆盖率(%)': coverage
            })
        
        # 生成波动率分析报告
        volatility_stats = {
            '波动率(标准差)': volatility,
            '波动率(%)': volatility_pct,
            '±0.1%噪声范围': f'{noise_lower:.4f} - {noise_upper:.4f}',
            '±0.1%覆盖率(%)': noise_coverage,
            '±1.2%覆盖率(%)': volatility_error_coverage,
            '样本总数': len(df)
        }
        
        # 保存波动率分析结果
        volatility_df = pd.DataFrame(list(volatility_stats.items()), columns=['指标', '值'])
        volatility_file = self.output_dir / 'volatility_analysis.csv'
        volatility_df.to_csv(volatility_file, index=False, encoding='utf-8-sig')
        
        # 保存误差分布
        error_df = pd.DataFrame(error_distribution)
        error_file = self.output_dir / 'error_distribution.csv'
        error_df.to_csv(error_file, index=False, encoding='utf-8-sig')
        
        logging.info(f"波动率分析已保存到: {volatility_file}")
        logging.info(f"误差分布已保存到: {error_file}")
        
        return volatility_stats, error_distribution
    
    def generate_noise_fill_strategies(self, volatility_stats, error_distribution):
        """生成基于波动率的噪声填充策略"""
        logging.info("生成噪声填充策略...")
        
        # 获取关键指标
        volatility = volatility_stats['波动率(标准差)']
        noise_coverage = volatility_stats['±0.1%覆盖率(%)']
        error_1_2_coverage = volatility_stats['±1.2%覆盖率(%)']
        
        # 生成策略
        noise_strategies = {
            '策略1_保守噪声': {
                '描述': '使用±0.1%的保守噪声范围',
                '公式': f'OpenPrice = PrevClosePrice * (1.0 + random(-0.001, 0.001))',
                '覆盖率': f'{noise_coverage:.2f}%',
                '适用场景': '需要最小偏差的精确填充',
                '优点': '偏差极小，符合大部分真实情况',
                '缺点': '可能不够随机，缺乏真实波动'
            },
            '策略2_标准噪声': {
                '描述': '使用1倍标准差的噪声范围',
                '公式': f'OpenPrice = PrevClosePrice * (1.0 + random(-{volatility:.4f}, {volatility:.4f}))',
                '覆盖率': '约68%',
                '适用场景': '平衡精度和真实性的标准填充',
                '优点': '符合正态分布，真实性强',
                '缺点': '偏差相对较大'
            },
            '策略3_安全噪声': {
                '描述': '使用±1.2%的安全噪声范围',
                '公式': 'OpenPrice = PrevClosePrice * (1.0 + random(-0.012, 0.012))',
                '覆盖率': f'{error_1_2_coverage:.2f}%',
                '适用场景': '需要覆盖大部分真实情况的填充',
                '优点': '覆盖率高，安全性好',
                '缺点': '偏差较大，可能不够精确'
            },
            '策略4_自适应噪声': {
                '描述': '根据板块特征调整噪声范围',
                '公式': 'OpenPrice = PrevClosePrice * (1.0 + random(-sector_vol, sector_vol))',
                '覆盖率': '板块特定',
                '适用场景': '不同板块波动特征差异较大时',
                '优点': '考虑板块特征，更精确',
                '缺点': '需要板块数据，计算复杂'
            },
            '策略5_混合策略': {
                '描述': '结合平均比率和噪声',
                '公式': f'OpenPrice = PrevClosePrice * ({self.get_average_ratio():.4f} + random(-{volatility:.4f}, {volatility:.4f}))',
                '覆盖率': '动态调整',
                '适用场景': '需要同时考虑趋势和波动的填充',
                '优点': '既考虑历史趋势，又保持随机性',
                '缺点': '计算复杂，需要历史数据'
            }
        }
        
        # 保存噪声填充策略
        noise_file = self.output_dir / 'noise_fill_strategies.md'
        with open(noise_file, 'w', encoding='utf-8') as f:
            f.write("# 基于波动率的噪声填充策略\n\n")
            f.write(f"基于 {self.start_date} 到 {self.end_date} 的数据分析\n\n")
            
            f.write("## 关键波动率指标\n\n")
            f.write(f"- 开盘价/前收价比率标准差: {volatility:.6f}\n")
            f.write(f"- 波动率百分比: {volatility*100:.4f}%\n")
            f.write(f"- ±0.1%噪声覆盖率: {noise_coverage:.2f}%\n")
            f.write(f"- ±1.2%误差覆盖率: {error_1_2_coverage:.2f}%\n\n")
            
            f.write("## 误差分布统计\n\n")
            f.write("| 误差范围 | 数据量 | 覆盖率 |\n")
            f.write("|---------|--------|--------|\n")
            for error_info in error_distribution:
                f.write(f"| {error_info['误差范围']} | {error_info['数据量']:,} | {error_info['覆盖率(%)']:.2f}% |\n")
            f.write("\n")
            
            f.write("## 推荐策略\n\n")
            f.write("### 1. 精确填充（推荐用于关键数据）\n")
            f.write(f"- 使用±0.1%噪声范围\n")
            f.write(f"- 覆盖率: {noise_coverage:.2f}%\n")
            f.write(f"- 偏差极小，适合精确计算\n\n")
            
            f.write("### 2. 标准填充（推荐用于一般数据）\n")
            f.write(f"- 使用1倍标准差噪声范围: ±{volatility*100:.4f}%\n")
            f.write(f"- 符合正态分布，真实性强\n\n")
            
            f.write("### 3. 安全填充（推荐用于风险控制）\n")
            f.write(f"- 使用±1.2%噪声范围\n")
            f.write(f"- 覆盖率: {error_1_2_coverage:.2f}%\n")
            f.write(f"- 覆盖大部分真实情况\n\n")
            
            f.write("## 详细策略说明\n\n")
            
            for strategy_name, strategy in noise_strategies.items():
                f.write(f"### {strategy_name}\n")
                f.write(f"- **描述**: {strategy['描述']}\n")
                f.write(f"- **公式**: {strategy['公式']}\n")
                f.write(f"- **覆盖率**: {strategy['覆盖率']}\n")
                f.write(f"- **适用场景**: {strategy['适用场景']}\n")
                f.write(f"- **优点**: {strategy['优点']}\n")
                f.write(f"- **缺点**: {strategy['缺点']}\n\n")
        
        logging.info(f"噪声填充策略已保存到: {noise_file}")
        return noise_strategies
    
    def get_average_ratio(self):
        """获取平均比率（用于混合策略）"""
        # 这里可以从之前的分析结果中获取，暂时返回1.0
        return 1.0
    
    def create_volatility_visualizations(self, df, error_distribution):
        """创建波动率相关的可视化图表"""
        logging.info("创建波动率可视化图表...")
        
        # 创建新的图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('开盘价波动率分析', fontsize=16, fontweight='bold')
        
        # 1. 波动率分布（对数收益率）
        axes[0, 0].hist(df['LogReturn'], bins=100, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].axvline(df['LogReturn'].mean(), color='red', linestyle='--', label=f'均值: {df["LogReturn"].mean():.6f}')
        axes[0, 0].axvline(df['LogReturn'].mean() + df['LogReturn'].std(), color='orange', linestyle='--', label=f'+1σ: {df["LogReturn"].mean() + df["LogReturn"].std():.6f}')
        axes[0, 0].axvline(df['LogReturn'].mean() - df['LogReturn'].std(), color='orange', linestyle='--', label=f'-1σ: {df["LogReturn"].mean() - df["LogReturn"].std():.6f}')
        axes[0, 0].set_xlabel('对数收益率')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('对数收益率分布（波动率分析）')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 误差范围覆盖率
        error_ranges = [info['误差范围'] for info in error_distribution]
        coverages = [info['覆盖率(%)'] for info in error_distribution]
        
        axes[0, 1].bar(error_ranges, coverages, color='lightgreen', alpha=0.7)
        axes[0, 1].set_xlabel('误差范围')
        axes[0, 1].set_ylabel('覆盖率 (%)')
        axes[0, 1].set_title('不同误差范围的数据覆盖率')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, coverage in enumerate(coverages):
            axes[0, 1].text(i, coverage + 1, f'{coverage:.1f}%', ha='center', va='bottom')
        
        # 3. 比率分布（突出显示噪声范围）
        axes[1, 0].hist(df['OpenCloseRatio'], bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
        
        # 标记不同的噪声范围
        axes[1, 0].axvspan(0.999, 1.001, alpha=0.3, color='green', label='±0.1%噪声范围')
        axes[1, 0].axvspan(0.988, 1.012, alpha=0.2, color='yellow', label='±1.2%误差范围')
        axes[1, 0].axvline(df['OpenCloseRatio'].mean(), color='red', linestyle='--', label=f'均值: {df["OpenCloseRatio"].mean():.4f}')
        
        axes[1, 0].set_xlabel('开盘价/前收价比率')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('比率分布（噪声范围标记）')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 累积分布函数
        sorted_ratios = np.sort(df['OpenCloseRatio'])
        cumulative_prob = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        
        axes[1, 1].plot(sorted_ratios, cumulative_prob, linewidth=2, color='blue')
        axes[1, 1].axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50%分位数')
        axes[1, 1].axhline(0.68, color='orange', linestyle='--', alpha=0.7, label='68%分位数')
        axes[1, 1].axhline(0.95, color='green', linestyle='--', alpha=0.7, label='95%分位数')
        
        axes[1, 1].set_xlabel('开盘价/前收价比率')
        axes[1, 1].set_ylabel('累积概率')
        axes[1, 1].set_title('比率累积分布函数')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        volatility_chart_file = self.output_dir / 'volatility_analysis.png'
        plt.savefig(volatility_chart_file, dpi=300, bbox_inches='tight')
        logging.info(f"波动率可视化图表已保存到: {volatility_chart_file}")
        
        plt.show()
    
    def analyze_sampling_statistics(self, df):
        """分析采样统计信息"""
        logging.info("分析采样统计信息...")
        
        if 'SampleYear' not in df.columns:
            logging.warning("数据中没有采样信息，跳过采样统计")
            return
        
        # 按年月统计采样数据
        sampling_stats = df.groupby(['SampleYear', 'SampleMonth']).agg({
            'SecuCode': 'count',
            'OpenCloseRatio': ['mean', 'std', 'min', 'max'],
            'OpenCloseRatioPct': ['mean', 'std', 'min', 'max']
        }).round(6)
        
        # 重命名列
        sampling_stats.columns = [
            '样本数', '平均比率', '比率标准差', '最小比率', '最大比率',
            '平均涨跌幅', '涨跌幅标准差', '最小涨跌幅', '最大涨跌幅'
        ]
        
        # 重置索引
        sampling_stats = sampling_stats.reset_index()
        
        # 添加年月标识
        sampling_stats['年月'] = sampling_stats['SampleYear'].astype(str) + '-' + sampling_stats['SampleMonth'].astype(str).str.zfill(2)
        
        # 保存采样统计
        sampling_file = self.output_dir / 'sampling_statistics.csv'
        sampling_stats.to_csv(sampling_file, index=False, encoding='utf-8-sig')
        logging.info(f"采样统计已保存到: {sampling_file}")
        
        # 计算总体采样统计
        total_samples = len(df)
        total_months = len(sampling_stats)
        avg_samples_per_month = total_samples / total_months if total_months > 0 else 0
        
        overall_sampling_stats = {
            '总采样数': total_samples,
            '采样月份数': total_months,
            '平均每月采样数': avg_samples_per_month,
            '采样时间范围': f"{df['SampleYear'].min()}-{df['SampleMonth'].min():02d} 到 {df['SampleYear'].max()}-{df['SampleMonth'].max():02d}",
            '采样股票数': df['SecuCode'].nunique(),
            '采样板块数': df['ListedSector'].nunique()
        }
        
        # 保存总体统计
        overall_file = self.output_dir / 'overall_sampling_stats.csv'
        overall_df = pd.DataFrame(list(overall_sampling_stats.items()), columns=['指标', '值'])
        overall_df.to_csv(overall_file, index=False, encoding='utf-8-sig')
        logging.info(f"总体采样统计已保存到: {overall_file}")
        
        return sampling_stats, overall_sampling_stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析开盘价与前一日收盘价的比率关系')
    parser.add_argument('--start', default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--sampling', action='store_true', default=True, help='使用随机采样（每月1000条）')
    parser.add_argument('--no-sampling', dest='sampling', action='store_false', help='使用全量数据')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = OpenCloseRatioAnalyzer(
        start_date=args.start,
        end_date=args.end
    )
    
    analyzer.run_analysis(use_sampling=args.sampling)


if __name__ == '__main__':
    main() 