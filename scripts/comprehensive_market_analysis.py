#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合市场数据分析脚本
分析价格差异、停牌复牌模式、成交量价格关系等
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
import warnings

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connector import JuyuanDB

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/logs/comprehensive_market_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ComprehensiveMarketAnalyzer:
    """综合市场数据分析器"""
    
    def __init__(self, year=2024, sample_size=1000):
        self.year = year
        self.sample_size = sample_size
        self.output_dir = Path(f'output/processed_data/comprehensive_analysis_{year}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 忽略字体警告
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
    def fetch_all_data(self):
        """获取所有数据（按月采样）"""
        logging.info(f"开始获取{self.year}年数据（每月采样{self.sample_size}条）")
        
        try:
            db = JuyuanDB()
            all_sampled_data = []
            
            for month in range(1, 13):
                start_date = f"{self.year}-{month:02d}-01"
                if month == 12:
                    end_date = f"{self.year}-12-31"
                else:
                    end_date = f"{self.year}-{month+1:02d}-01"
                    end_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
                
                logging.info(f"采样 {self.year}年{month}月 数据 ({start_date} 到 {end_date})")
                
                # 查询单月数据
                sql = f"""
                SELECT 
                    c.SecuCode,
                    c.SecuMarket,
                    c.ListedSector,
                    a.TradingDay,
                    a.OpenPrice,
                    a.ClosePrice,
                    a.HighPrice,
                    a.LowPrice,
                    a.TurnoverVolume as Volume,
                    a.TurnoverValue as Amount,
                    a.PrevClosePrice,
                    d.Ifsuspend,
                    s.SuspendReason,
                    s.ResumptionDate as ResumeDate
                FROM QT_DailyQuote a
                LEFT JOIN SecuMain c ON a.InnerCode = c.InnerCode
                LEFT JOIN QT_StockPerformance d ON a.InnerCode = d.InnerCode AND a.TradingDay = d.TradingDay
                LEFT JOIN LC_SuspendResumption s ON a.InnerCode = s.InnerCode AND a.TradingDay = s.SuspendDate
                WHERE c.SecuCategory = 1 
                    AND c.ListedState = 1
                    AND c.SecuMarket IN (83, 90)
                    AND a.TradingDay BETWEEN '{start_date}' AND '{end_date}'
                    AND a.OpenPrice IS NOT NULL
                    AND a.ClosePrice IS NOT NULL
                    AND a.HighPrice IS NOT NULL
                    AND a.LowPrice IS NOT NULL
                ORDER BY a.TradingDay, c.SecuCode
                """
                
                month_df = db.read_sql(sql)
                
                if not month_df.empty:
                    # 随机采样
                    sample_size = min(self.sample_size, len(month_df))
                    if len(month_df) > sample_size:
                        sampled_df = month_df.sample(n=sample_size, random_state=42)
                        logging.info(f"{self.year}年{month}月: 总数据{len(month_df)}行，随机采样{sample_size}行")
                    else:
                        sampled_df = month_df
                        logging.info(f"{self.year}年{month}月: 总数据{len(month_df)}行，全部采样")
                    
                    # 添加采样标识
                    sampled_df['SampleYear'] = self.year
                    sampled_df['SampleMonth'] = month
                    sampled_df['SampleType'] = 'Random'
                    
                    all_sampled_data.append(sampled_df)
                else:
                    logging.warning(f"{self.year}年{month}月: 无数据")
            
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
            logging.error(f"获取数据失败: {str(e)}")
            return pd.DataFrame()
    
    def analyze_price_gaps(self, df):
        """分析价格差异"""
        logging.info("分析价格差异...")
        
        # 过滤有效数据
        valid_df = df[
            (df['OpenPrice'] > 0) & 
            (df['ClosePrice'] > 0) & 
            (df['HighPrice'] > 0) & 
            (df['LowPrice'] > 0) & 
            (df['Ifsuspend'] != 1)  # 排除停牌数据
        ].copy()
        
        if valid_df.empty:
            logging.warning("无有效数据用于价格差异分析")
            return pd.DataFrame(), {}
        
        # 计算价格差异
        valid_df['MaxOpenClose'] = valid_df[['OpenPrice', 'ClosePrice']].max(axis=1)
        valid_df['MinOpenClose'] = valid_df[['OpenPrice', 'ClosePrice']].min(axis=1)
        valid_df['HighGap'] = valid_df['HighPrice'] - valid_df['MaxOpenClose']
        valid_df['LowGap'] = valid_df['MinOpenClose'] - valid_df['LowPrice']
        
        # 计算百分比差异
        valid_df['HighGapPct'] = (valid_df['HighGap'] / valid_df['MaxOpenClose']) * 100
        valid_df['LowGapPct'] = (valid_df['LowGap'] / valid_df['MinOpenClose']) * 100
        
        # 计算成交额调整因子
        valid_df['VolumeCloseValue'] = valid_df['Volume'] * valid_df['ClosePrice']  # 成交量×收盘价
        valid_df['AmountAdjustFactor'] = (valid_df['Amount'] - valid_df['VolumeCloseValue']) / valid_df['VolumeCloseValue'] * 100  # 调整因子百分比
        valid_df['AvgPrice'] = valid_df['Amount'] / valid_df['Volume']  # 实际成交均价
        valid_df['AvgPriceAdjustFactor'] = (valid_df['AvgPrice'] - valid_df['ClosePrice']) / valid_df['ClosePrice'] * 100  # 均价调整因子百分比
        
        # 计算开盘噪音
        valid_df['OpenCloseRatio'] = valid_df['OpenPrice'] / valid_df['PrevClosePrice']
        valid_df['OpenCloseNoise'] = (valid_df['OpenCloseRatio'] - 1) * 100
        
        # 按月统计
        monthly_stats = []
        for month in range(1, 13):
            month_data = valid_df[valid_df['SampleMonth'] == month]
            if len(month_data) > 0:
                stats = {
                    'SampleMonth': month,
                    '样本数': len(month_data),
                    '最高价差异_均值': month_data['HighGap'].mean(),
                    '最高价差异_标准差': month_data['HighGap'].std(),
                    '最高价差异百分比_均值': month_data['HighGapPct'].mean(),
                    '最高价差异百分比_标准差': month_data['HighGapPct'].std(),
                    '最低价差异_均值': month_data['LowGap'].mean(),
                    '最低价差异_标准差': month_data['LowGap'].std(),
                    '最低价差异百分比_均值': month_data['LowGapPct'].mean(),
                    '最低价差异百分比_标准差': month_data['LowGapPct'].std(),
                    '成交额调整因子_均值': month_data['AmountAdjustFactor'].mean(),
                    '成交额调整因子_标准差': month_data['AmountAdjustFactor'].std(),
                    '均价调整因子_均值': month_data['AvgPriceAdjustFactor'].mean(),
                    '均价调整因子_标准差': month_data['AvgPriceAdjustFactor'].std(),
                    '开盘噪音_均值': month_data['OpenCloseNoise'].mean(),
                    '开盘噪音_标准差': month_data['OpenCloseNoise'].std(),
                    '开盘噪音_中位数': month_data['OpenCloseNoise'].median()
                }
                monthly_stats.append(stats)
        
        monthly_df = pd.DataFrame(monthly_stats)
        
        # 总体统计
        overall_stats = {
            '样本数': len(valid_df),
            '最高价差异_均值': valid_df['HighGap'].mean(),
            '最高价差异_标准差': valid_df['HighGap'].std(),
            '最高价差异百分比_均值': valid_df['HighGapPct'].mean(),
            '最高价差异百分比_标准差': valid_df['HighGapPct'].std(),
            '最低价差异_均值': valid_df['LowGap'].mean(),
            '最低价差异_标准差': valid_df['LowGap'].std(),
            '最低价差异百分比_均值': valid_df['LowGapPct'].mean(),
            '最低价差异百分比_标准差': valid_df['LowGapPct'].std(),
            '成交额调整因子_均值': valid_df['AmountAdjustFactor'].mean(),
            '成交额调整因子_标准差': valid_df['AmountAdjustFactor'].std(),
            '均价调整因子_均值': valid_df['AvgPriceAdjustFactor'].mean(),
            '均价调整因子_标准差': valid_df['AvgPriceAdjustFactor'].std(),
            '开盘噪音_均值': valid_df['OpenCloseNoise'].mean(),
            '开盘噪音_标准差': valid_df['OpenCloseNoise'].std(),
            '开盘噪音_中位数': valid_df['OpenCloseNoise'].median()
        }
        
        # 保存结果
        monthly_file = self.output_dir / 'price_gaps_monthly.csv'
        monthly_df.to_csv(monthly_file, index=False, encoding='utf-8-sig')
        
        overall_file = self.output_dir / 'price_gaps_overall.csv'
        overall_df = pd.DataFrame(list(overall_stats.items()), columns=['指标', '值'])
        overall_df.to_csv(overall_file, index=False, encoding='utf-8-sig')
        
        logging.info(f"价格差异分析已保存到: {monthly_file} 和 {overall_file}")
        
        return monthly_df, overall_stats
    
    def analyze_suspend_resume_patterns(self, df):
        """分析停牌复牌模式（基于中国交易日历）"""
        logging.info("分析停牌复牌模式（基于交易日历）...")
        
        # 过滤停牌数据
        suspend_df = df[df['Ifsuspend'] == 1].copy()
        
        if suspend_df.empty:
            logging.warning("无停牌数据")
            return pd.DataFrame(), pd.DataFrame()
        
        # 分析复牌时间（基于交易日历）
        resume_patterns = []
        db = JuyuanDB()
        
        for _, row in suspend_df.iterrows():
            if pd.notna(row['ResumeDate']) and pd.notna(row['TradingDay']):
                try:
                    suspend_date = pd.to_datetime(row['TradingDay'])
                    resume_date = pd.to_datetime(row['ResumeDate'])
                    
                    # 使用交易日历计算停牌天数
                    suspend_trading_days = self.get_trading_days_between(
                        suspend_date.strftime('%Y-%m-%d'), 
                        resume_date.strftime('%Y-%m-%d'), 
                        db
                    )
                    
                    if suspend_trading_days >= 0:
                        resume_patterns.append({
                            'SecuCode': row['SecuCode'],
                            'TradingDay': row['TradingDay'],
                            'ResumeDate': row['ResumeDate'],
                            'SuspendTradingDays': suspend_trading_days,  # 交易日天数
                            'SuspendNaturalDays': (resume_date - suspend_date).days,  # 自然日天数
                            'SuspendReason': row.get('SuspendReason', ''),
                            'SampleMonth': row['SampleMonth']
                        })
                except Exception as e:
                    logging.warning(f"计算停牌天数失败: {row['SecuCode']} {row['TradingDay']} - {str(e)}")
                    continue
        
        db.close()
        
        if not resume_patterns:
            logging.warning("无法计算停牌天数")
            return pd.DataFrame(), pd.DataFrame()
        
        patterns_df = pd.DataFrame(resume_patterns)
        
        # 统计停牌交易日分布
        suspend_distribution = patterns_df['SuspendTradingDays'].value_counts().sort_index()
        distribution_stats = []
        
        for days, count in suspend_distribution.items():
            distribution_stats.append({
                '停牌交易日数': days,
                '股票数': count,
                '占比(%)': count / len(patterns_df) * 100
            })
        
        distribution_df = pd.DataFrame(distribution_stats)
        
        # 按月统计停牌情况
        monthly_suspend = patterns_df.groupby('SampleMonth').agg({
            'SuspendTradingDays': ['count', 'mean', 'std', 'min', 'max'],
            'SuspendNaturalDays': ['mean', 'std']
        }).round(2)
        
        # 重命名列
        monthly_suspend.columns = [
            '停牌事件数', '平均停牌交易日', '停牌交易日标准差', '最短停牌交易日', '最长停牌交易日',
            '平均停牌自然日', '停牌自然日标准差'
        ]
        monthly_suspend = monthly_suspend.reset_index()
        
        # 保存结果
        patterns_file = self.output_dir / 'suspend_patterns.csv'
        patterns_df.to_csv(patterns_file, index=False, encoding='utf-8-sig')
        
        distribution_file = self.output_dir / 'resume_distribution.csv'
        distribution_df.to_csv(distribution_file, index=False, encoding='utf-8-sig')
        
        monthly_file = self.output_dir / 'monthly_suspend_stats.csv'
        monthly_suspend.to_csv(monthly_file, index=False, encoding='utf-8-sig')
        
        logging.info(f"停牌复牌分析已保存到: {patterns_file}, {distribution_file}, {monthly_file}")
        
        return distribution_df, monthly_suspend
    
    def get_trading_days_between(self, start_date, end_date, db):
        """计算两个日期之间的交易日数量
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            db: 数据库连接
            
        Returns:
            int: 交易日数量
        """
        try:
            # 使用QT_TradingDayNew表计算交易日数量
            sql = f"""
            SELECT COUNT(*) as trading_days
            FROM QT_TradingDayNew
            WHERE TradingDate BETWEEN '{start_date}' AND '{end_date}'
                AND SecuMarket IN (83, 90)  -- 中国股市（深交所、上交所）
                AND IfTradingDay = 1  -- 是交易日
            """
            result = db.read_sql(sql)
            trading_days = result.iloc[0]['trading_days'] if not result.empty else 0
            
            # 添加调试日志
            natural_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
            logging.debug(f"交易日计算: {start_date} 到 {end_date}")
            logging.debug(f"  自然日数: {natural_days}")
            logging.debug(f"  交易日数: {trading_days}")
            logging.debug(f"  差异(节假日+周末): {natural_days - trading_days}")
            
            return trading_days
            
        except Exception as e:
            logging.warning(f"计算交易日数量失败: {str(e)}，使用自然日计算")
            # 回退到自然日计算
            natural_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
            return natural_days
    
    def analyze_volume_price_relationship(self, df):
        """分析成交量价格关系"""
        logging.info("分析成交量价格关系...")
        
        # 过滤有效数据
        valid_df = df[
            (df['Volume'] > 0) & 
            (df['OpenPrice'] > 0) & 
            (df['ClosePrice'] > 0) & 
            (df['Ifsuspend'] != 1)
        ].copy()
        
        if valid_df.empty:
            logging.warning("无有效数据用于成交量价格分析")
            return pd.DataFrame(), {}
        
        # 计算价格变化
        valid_df['PriceChange'] = valid_df['ClosePrice'] - valid_df['OpenPrice']
        valid_df['PriceChangePct'] = (valid_df['PriceChange'] / valid_df['OpenPrice']) * 100
        
        # 计算成交量变化（如果有前一日数据）
        if 'PrevClosePrice' in valid_df.columns:
            valid_df['VolumeChange'] = valid_df['Volume'] - valid_df['Volume'].shift(1)
            valid_df['VolumeChangePct'] = (valid_df['VolumeChange'] / valid_df['Volume'].shift(1)) * 100
        else:
            valid_df['VolumeChangePct'] = 0
        
        # 按月统计
        monthly_stats = []
        for month in range(1, 13):
            month_data = valid_df[valid_df['SampleMonth'] == month]
            if len(month_data) > 0:
                stats = {
                    'SampleMonth': month,
                    '样本数': len(month_data),
                    '成交量_均值': month_data['Volume'].mean(),
                    '成交量_标准差': month_data['Volume'].std(),
                    '成交量_中位数': month_data['Volume'].median(),
                    '价格变化百分比_均值': month_data['PriceChangePct'].mean(),
                    '价格变化百分比_标准差': month_data['PriceChangePct'].std(),
                    '价格变化百分比_中位数': month_data['PriceChangePct'].median(),
                    '成交量变化百分比_均值': month_data['VolumeChangePct'].mean(),
                    '成交量变化百分比_标准差': month_data['VolumeChangePct'].std()
                }
                monthly_stats.append(stats)
        
        monthly_df = pd.DataFrame(monthly_stats)
        
        # 总体统计
        overall_stats = {
            '样本数': len(valid_df),
            '成交量_均值': valid_df['Volume'].mean(),
            '成交量_标准差': valid_df['Volume'].std(),
            '成交量_中位数': valid_df['Volume'].median(),
            '价格变化百分比_均值': valid_df['PriceChangePct'].mean(),
            '价格变化百分比_标准差': valid_df['PriceChangePct'].std(),
            '价格变化百分比_中位数': valid_df['PriceChangePct'].median(),
            '成交量变化百分比_均值': valid_df['VolumeChangePct'].mean(),
            '成交量变化百分比_标准差': valid_df['VolumeChangePct'].std()
        }
        
        # 保存结果
        monthly_file = self.output_dir / 'volume_price_monthly.csv'
        monthly_df.to_csv(monthly_file, index=False, encoding='utf-8-sig')
        
        overall_file = self.output_dir / 'volume_price_overall.csv'
        overall_df = pd.DataFrame(list(overall_stats.items()), columns=['指标', '值'])
        overall_df.to_csv(overall_file, index=False, encoding='utf-8-sig')
        
        logging.info(f"成交量价格分析已保存到: {monthly_file} 和 {overall_file}")
        
        return monthly_df, overall_stats
    
    def create_visualizations(self, price_gaps_monthly, resume_distribution, volume_price_monthly):
        """创建可视化图表"""
        logging.info("创建可视化图表...")
        
        # 设置图表样式
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Comprehensive Market Analysis {self.year}', fontsize=16, fontweight='bold')
        
        # 1. 价格差异分析
        if not price_gaps_monthly.empty:
            months = price_gaps_monthly['SampleMonth']
            
            axes[0, 0].plot(months, price_gaps_monthly['最高价差异百分比_均值'], 'o-', label='High Price Gap', color='red')
            axes[0, 0].plot(months, price_gaps_monthly['最低价差异百分比_均值'], 's-', label='Low Price Gap', color='blue')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('Gap Percentage (%)')
            axes[0, 0].set_title('Price Gap Monthly Trend')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xticks(range(1, 13))
        
        # 2. 开盘噪音分析
        if not price_gaps_monthly.empty:
            axes[0, 1].plot(months, price_gaps_monthly['开盘噪音_均值'], 'o-', color='green', label='Opening Noise Mean')
            axes[0, 1].fill_between(months, 
                                   price_gaps_monthly['开盘噪音_均值'] - price_gaps_monthly['开盘噪音_标准差'],
                                   price_gaps_monthly['开盘噪音_均值'] + price_gaps_monthly['开盘噪音_标准差'],
                                   alpha=0.3, color='green', label='±1 Std Dev')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Opening Noise (%)')
            axes[0, 1].set_title('Opening Price Noise Monthly Trend')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xticks(range(1, 13))
        
        # 3. 停牌复牌分布
        if not resume_distribution.empty:
            days = resume_distribution['停牌交易日数']
            counts = resume_distribution['股票数']
            
            axes[1, 0].bar(days, counts, color='orange', alpha=0.7)
            axes[1, 0].set_xlabel('Suspension Trading Days')
            axes[1, 0].set_ylabel('Number of Stocks')
            axes[1, 0].set_title('Suspension Duration Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (day, count) in enumerate(zip(days, counts)):
                axes[1, 0].text(day, count + max(counts)*0.01, f'{count}', ha='center', va='bottom')
        
        # 4. 成交量价格关系
        if not volume_price_monthly.empty:
            months = volume_price_monthly['SampleMonth']
            
            ax1 = axes[1, 1]
            ax2 = ax1.twinx()
            
            # 成交量变化
            line1 = ax1.plot(months, volume_price_monthly['成交量变化百分比_均值'], 'o-', color='purple', label='Volume Change')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Volume Change (%)', color='purple')
            ax1.tick_params(axis='y', labelcolor='purple')
            
            # 价格变化
            line2 = ax2.plot(months, volume_price_monthly['价格变化百分比_均值'], 's-', color='brown', label='Price Change')
            ax2.set_ylabel('Price Change (%)', color='brown')
            ax2.tick_params(axis='y', labelcolor='brown')
            
            ax1.set_title('Volume vs Price Change Relationship')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(range(1, 13))
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.output_dir / 'comprehensive_analysis.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        logging.info(f"可视化图表已保存到: {chart_file}")
        
        plt.show()
    
    def run_analysis(self):
        """运行完整分析"""
        logging.info(f"开始{self.year}年综合市场数据分析")
        
        # 1. 获取数据
        df = self.fetch_all_data()
        if df.empty:
            logging.error("无法获取数据，分析终止")
            return
        
        # 2. 分析价格差异
        price_gaps_monthly, price_gaps_overall = self.analyze_price_gaps(df)
        
        # 3. 分析停牌复牌模式
        resume_distribution, monthly_suspend = self.analyze_suspend_resume_patterns(df)
        
        # 4. 分析成交量价格关系
        volume_price_monthly, volume_price_overall = self.analyze_volume_price_relationship(df)
        
        # 5. 创建可视化
        self.create_visualizations(price_gaps_monthly, resume_distribution, volume_price_monthly)
        
        # 6. 输出总结
        self.print_summary(price_gaps_overall, resume_distribution, volume_price_overall)
        
        logging.info("分析完成")
    
    def print_summary(self, price_gaps_overall, resume_distribution, volume_price_overall):
        """打印分析总结"""
        print("\n" + "="*80)
        print(f"{self.year}年综合市场数据分析总结")
        print("="*80)
        
        print(f"\n📊 价格差异分析:")
        print(f"  最高价差异: 平均 {price_gaps_overall['最高价差异百分比_均值']:.2f}%")
        print(f"  最低价差异: 平均 {price_gaps_overall['最低价差异百分比_均值']:.2f}%")
        print(f"  开盘噪音: 平均 {price_gaps_overall['开盘噪音_均值']:.2f}% (标准差: {price_gaps_overall['开盘噪音_标准差']:.2f}%)")
        print(f"  成交额调整因子: 平均 {price_gaps_overall['成交额调整因子_均值']:.2f}% (标准差: {price_gaps_overall['成交额调整因子_标准差']:.2f}%)")
        print(f"  均价调整因子: 平均 {price_gaps_overall['均价调整因子_均值']:.2f}% (标准差: {price_gaps_overall['均价调整因子_标准差']:.2f}%)")
        
        print(f"\n📈 停牌复牌分析:")
        if not resume_distribution.empty:
            total_suspend = resume_distribution['股票数'].sum()
            one_day = resume_distribution[resume_distribution['停牌交易日数'] == 1]['股票数'].iloc[0] if len(resume_distribution[resume_distribution['停牌交易日数'] == 1]) > 0 else 0
            two_day = resume_distribution[resume_distribution['停牌交易日数'] == 2]['股票数'].iloc[0] if len(resume_distribution[resume_distribution['停牌交易日数'] == 2]) > 0 else 0
            
            print(f"  总停牌事件: {total_suspend} 次")
            print(f"  第一天复牌: {one_day} 次 ({one_day/total_suspend*100:.1f}%)")
            print(f"  第二天复牌: {two_day} 次 ({two_day/total_suspend*100:.1f}%)")
            print(f"  其他天数复牌: {total_suspend - one_day - two_day} 次 ({(total_suspend - one_day - two_day)/total_suspend*100:.1f}%)")
        else:
            print("  无停牌数据")
        
        print(f"\n💰 成交量价格分析:")
        print(f"  平均成交量: {volume_price_overall['成交量_均值']:.0f}")
        print(f"  平均价格变化: {volume_price_overall['价格变化百分比_均值']:.2f}% (标准差: {volume_price_overall['价格变化百分比_标准差']:.2f}%)")
        
        print(f"\n📁 结果文件:")
        print(f"  数据目录: {self.output_dir}")
        print(f"  分析图表: {self.output_dir / 'comprehensive_analysis.png'}")
        
        print("="*80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='综合市场数据分析')
    parser.add_argument('--year', type=int, default=2024, help='分析年份')
    parser.add_argument('--sample-size', type=int, default=1000, help='每月采样数量')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = ComprehensiveMarketAnalyzer(
        year=args.year,
        sample_size=args.sample_size
    )
    
    analyzer.run_analysis()


if __name__ == '__main__':
    main()