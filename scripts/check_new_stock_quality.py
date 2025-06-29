#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新股上市数据质量专项检查脚本
专门检查新股上市后的特殊情况，包括涨跌停限制、上市状态等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connector import JuyuanDB

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/logs/new_stock_quality_check.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class NewStockQualityChecker:
    """新股上市数据质量检查器"""
    
    def __init__(self, start_date='2024-01-01', end_date='2024-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path('output/processed_data/new_stock_quality')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 板块涨跌停限制配置
        self.limit_configs = {
            83: {'name': '深交所主板', 'limit': 0.10},  # 深交所主板
            90: {'name': '上交所主板', 'limit': 0.10},  # 上交所主板
        }
        
    def get_new_stock_data(self):
        """获取新股上市数据"""
        logging.info("获取新股上市数据...")
        
        try:
            db = JuyuanDB()
            
            # 查询新股上市数据
            sql = f"""
            SELECT 
                c.SecuCode,
                c.SecuMarket,
                c.SecuCategory,
                c.ListedDate,
                c.ListedState,
                c.ListedSector,
                c.ChiName,
                c.SecuAbbr,
                a.TradingDay,
                a.PrevClosePrice,
                a.OpenPrice,
                a.HighPrice,
                a.LowPrice,
                a.ClosePrice,
                a.TurnoverVolume,
                a.TurnoverValue,
                b.PriceCeiling,
                b.PriceFloor,
                d.Ifsuspend
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
                AND c.ListedDate IS NOT NULL
                AND c.ListedDate BETWEEN '{self.start_date}' AND '{self.end_date}'
                AND a.TradingDay BETWEEN c.ListedDate AND '{self.end_date}'
            ORDER BY c.ListedDate, c.SecuCode, a.TradingDay
            """
            
            df = db.read_sql(sql)
            db.close()
            
            logging.info(f"获取新股数据成功，共 {len(df)} 行")
            return df
            
        except Exception as e:
            logging.error(f"获取新股数据失败: {str(e)}")
            return pd.DataFrame()
    
    def analyze_new_stock_quality(self, df):
        """分析新股数据质量"""
        if df.empty:
            logging.warning("无新股数据")
            return pd.DataFrame(), pd.DataFrame()
        
        logging.info("开始分析新股数据质量...")
        
        quality_issues = []
        problem_rows = []
        
        # 按股票分组分析
        for secu_code in df['SecuCode'].unique():
            stock_data = df[df['SecuCode'] == secu_code].copy()
            stock_data = stock_data.sort_values('TradingDay')
            
            # 获取股票基本信息
            listed_date = stock_data['ListedDate'].iloc[0]
            listed_sector = stock_data['ListedSector'].iloc[0]
            secu_market = stock_data['SecuMarket'].iloc[0]
            
            logging.info(f"分析股票 {secu_code} (上市日期: {listed_date}, 板块: {listed_sector})")
            
            # 分析每个交易日
            for idx, row in stock_data.iterrows():
                issues = []
                trading_day = row['TradingDay']
                
                # 计算上市天数
                days_since_listed = (trading_day - listed_date).days
                
                # 检查上市日期合理性
                if days_since_listed < 0:
                    issues.append(f"上市日期异常:交易日早于上市日期({days_since_listed}天)")
                
                # 检查涨跌停限制
                self.check_price_limit_issues(row, days_since_listed, listed_sector, issues)
                
                # 检查价格数据合理性
                self.check_price_data_issues(row, days_since_listed, issues)
                
                # 检查成交量数据
                self.check_volume_data_issues(row, days_since_listed, issues)
                
                # 检查上市状态
                self.check_listing_status_issues(row, issues)
                
                if issues:
                    # 添加到问题分析列表
                    issue_row = {
                        'SecuCode': secu_code,
                        'SecuMarket': secu_market,
                        'ListedSector': listed_sector,
                        'ListedDate': listed_date,
                        'TradingDay': trading_day,
                        'DaysSinceListed': days_since_listed,
                        'Issues': '; '.join(issues),
                        'IssueCount': len(issues)
                    }
                    
                    # 添加价格数据
                    for field in ['PrevClosePrice', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'PriceCeiling', 'PriceFloor']:
                        issue_row[f'{field}_Value'] = row.get(field)
                    
                    quality_issues.append(issue_row)
                    
                    # 添加到原始问题数据
                    problem_row = row.copy()
                    problem_row['Issues'] = '; '.join(issues)
                    problem_row['IssueCount'] = len(issues)
                    problem_row['DaysSinceListed'] = days_since_listed
                    problem_rows.append(problem_row)
        
        return pd.DataFrame(quality_issues), pd.DataFrame(problem_rows)
    
    def check_price_limit_issues(self, row, days_since_listed, listed_sector, issues):
        """检查涨跌停限制问题"""
        price_ceiling = row.get('PriceCeiling')
        price_floor = row.get('PriceFloor')
        prev_close = row.get('PrevClosePrice', 0)
        
        # 前5个交易日不设涨跌幅限制
        if days_since_listed <= 5:
            if price_ceiling is not None and price_ceiling != 0:
                issues.append(f"涨跌停限制:上市{days_since_listed}天应无涨跌停限制(PriceCeiling={price_ceiling})")
            if price_floor is not None and price_floor != 0:
                issues.append(f"涨跌停限制:上市{days_since_listed}天应无涨跌停限制(PriceFloor={price_floor})")
        else:
            # 第6个交易日起应有涨跌停限制
            if price_ceiling is None or price_ceiling == 0:
                issues.append(f"涨跌停限制:上市{days_since_listed}天应设涨跌停限制(PriceCeiling为空)")
            if price_floor is None or price_floor == 0:
                issues.append(f"涨跌停限制:上市{days_since_listed}天应设涨跌停限制(PriceFloor为空)")
            
            # 检查涨跌停幅度
            if price_ceiling is not None and price_ceiling > 0 and prev_close > 0:
                limit_ratio = (price_ceiling - prev_close) / prev_close
                expected_ratio = self.limit_configs.get(listed_sector, {}).get('limit', 0.10)
                
                if abs(limit_ratio - expected_ratio) > 0.001:
                    issues.append(f"涨跌停幅度:实际{limit_ratio:.3f}, 预期{expected_ratio:.3f}")
    
    def check_price_data_issues(self, row, days_since_listed, issues):
        """检查价格数据问题"""
        # 检查价格字段的合理性
        for field in ['PrevClosePrice', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice']:
            value = row.get(field)
            
            if pd.isna(value):
                issues.append(f"{field}:NULL/NaN")
            elif value < 0:
                issues.append(f"{field}:负价格({value})")
            elif value == 0 and days_since_listed > 0:  # 上市首日可以为0
                issues.append(f"{field}:价格为零(非上市首日)")
        
        # 检查价格逻辑关系
        if not pd.isna(row['HighPrice']) and not pd.isna(row['LowPrice']) and not pd.isna(row['ClosePrice']):
            if row['HighPrice'] < row['LowPrice']:
                issues.append("价格逻辑:最高价小于最低价")
            if row['ClosePrice'] > row['HighPrice']:
                issues.append("价格逻辑:收盘价高于最高价")
            if row['ClosePrice'] < row['LowPrice']:
                issues.append("价格逻辑:收盘价低于最低价")
    
    def check_volume_data_issues(self, row, days_since_listed, issues):
        """检查成交量数据问题"""
        volume = row.get('TurnoverVolume')
        value = row.get('TurnoverValue')
        
        if pd.isna(volume):
            issues.append("TurnoverVolume:NULL/NaN")
        elif volume < 0:
            issues.append(f"TurnoverVolume:负成交量({volume})")
        
        if pd.isna(value):
            issues.append("TurnoverValue:NULL/NaN")
        elif value < 0:
            issues.append(f"TurnoverValue:负成交额({value})")
    
    def check_listing_status_issues(self, row, issues):
        """检查上市状态问题"""
        listed_state = row.get('ListedState')
        if listed_state is None:
            issues.append("上市状态:缺少上市状态信息")
        elif listed_state not in [1, 2, 3]:
            issues.append(f"上市状态:状态值异常({listed_state})")
    
    def generate_new_stock_report(self, quality_issues, problem_rows):
        """生成新股质量报告"""
        logging.info("生成新股质量报告...")
        
        # 统计信息
        total_new_stocks = len(quality_issues['SecuCode'].unique()) if not quality_issues.empty else 0
        total_issue_rows = len(quality_issues)
        
        # 按问题类型统计
        issue_type_counts = {}
        if not quality_issues.empty:
            for _, row in quality_issues.iterrows():
                issues = row['Issues'].split('; ')
                for issue in issues:
                    issue_type = issue.split(':')[0] if ':' in issue else issue
                    issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
        
        # 按上市天数统计
        days_issue_stats = {}
        if not quality_issues.empty:
            for _, row in quality_issues.iterrows():
                days = row['DaysSinceListed']
                if days not in days_issue_stats:
                    days_issue_stats[days] = 0
                days_issue_stats[days] += 1
        
        # 生成报告内容
        report_content = f"""
新股上市数据质量检查报告
========================

检查时间范围: {self.start_date} 到 {self.end_date}
检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

检查项目:
- 上市日期合理性检查
- 涨跌停限制检查 (前5天无限制，第6天起有限制)
- 价格数据合理性检查
- 成交量数据检查
- 上市状态检查

总体统计:
- 新股数量: {total_new_stocks}
- 问题行数: {total_issue_rows}

问题类型分布:
"""
        
        for issue_type, count in sorted(issue_type_counts.items(), key=lambda x: x[1], reverse=True):
            report_content += f"- {issue_type}: {count:,} 次\n"
        
        report_content += "\n按上市天数统计:\n"
        for days in sorted(days_issue_stats.keys()):
            count = days_issue_stats[days]
            report_content += f"- 上市{days}天: {count:,} 行问题\n"
        
        # 保存报告
        report_file = self.output_dir / 'new_stock_quality_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 保存问题数据
        if not quality_issues.empty:
            quality_issues.to_csv(self.output_dir / 'new_stock_quality_issues.csv', 
                                index=False, encoding='utf-8-sig')
        
        if not problem_rows.empty:
            problem_rows.to_csv(self.output_dir / 'new_stock_problem_rows.csv', 
                              index=False, encoding='utf-8-sig')
        
        logging.info(f"新股质量报告已保存到: {report_file}")
        
        # 打印关键统计
        print(f"\n=== 新股数据质量检查完成 ===")
        print(f"新股数量: {total_new_stocks}")
        print(f"问题行数: {total_issue_rows}")
        print(f"详细结果保存在: {self.output_dir}")
        
        return report_content
    
    def run_check(self):
        """运行新股数据质量检查"""
        logging.info("开始新股上市数据质量检查")
        
        # 获取新股数据
        df = self.get_new_stock_data()
        
        if df.empty:
            logging.warning("无新股数据")
            return
        
        # 分析数据质量
        quality_issues, problem_rows = self.analyze_new_stock_quality(df)
        
        # 生成报告
        self.generate_new_stock_report(quality_issues, problem_rows)
        
        logging.info("新股数据质量检查完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='新股上市数据质量检查')
    parser.add_argument('--start', default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='结束日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # 创建检查器并运行
    checker = NewStockQualityChecker(
        start_date=args.start,
        end_date=args.end
    )
    
    checker.run_check()


if __name__ == '__main__':
    main() 