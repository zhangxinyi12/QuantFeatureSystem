#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试交易日计算功能 - 验证中国股市交易日历 (QT_TradingDayNew表)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connector import JuyuanDB
import pandas as pd
from datetime import datetime, timedelta

def test_trading_days_calculation():
    """测试交易日计算"""
    print("="*60)
    print("测试中国股市交易日计算功能 (QT_TradingDayNew表)")
    print("="*60)
    
    try:
        db = JuyuanDB()
        
        # 1. 检查交易日历表结构
        print("\n1. 检查QT_TradingDayNew表结构:")
        try:
            structure_sql = """
            SELECT TOP 5 *
            FROM QT_TradingDayNew
            WHERE SecuMarket IN (83, 90)
            ORDER BY TradingDate
            """
            structure_df = db.read_sql(structure_sql)
            print(f"交易日历表字段: {list(structure_df.columns)}")
            print(f"前5行数据:")
            for _, row in structure_df.iterrows():
                print(f"  {row['TradingDate']} - 市场{row['SecuMarket']} - 交易日{row['IfTradingDay']}")
        except Exception as e:
            print(f"检查表结构失败: {str(e)}")
        
        # 2. 测试几个关键日期范围
        print("\n2. 测试关键日期范围的交易日计算:")
        test_cases = [
            ('2024-01-01', '2024-01-05', '新年假期'),
            ('2024-02-10', '2024-02-16', '春节假期'),
            ('2024-05-01', '2024-05-05', '劳动节假期'),
            ('2024-10-01', '2024-10-07', '国庆假期'),
            ('2024-01-01', '2024-01-31', '整月测试'),
        ]
        
        for start_date, end_date, description in test_cases:
            print(f"\n{description} ({start_date} 到 {end_date}):")
            
            # 计算交易日
            sql = f"""
            SELECT COUNT(*) as trading_days
            FROM QT_TradingDayNew
            WHERE TradingDate BETWEEN '{start_date}' AND '{end_date}'
                AND SecuMarket IN (83, 90)  -- 中国股市
                AND IfTradingDay = 1  -- 是交易日
            """
            result = db.read_sql(sql)
            trading_days = result.iloc[0]['trading_days'] if not result.empty else 0
            
            # 计算自然日
            natural_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
            
            print(f"  自然日数: {natural_days}")
            print(f"  交易日数: {trading_days}")
            print(f"  差异(节假日+周末): {natural_days - trading_days}")
            
            # 显示具体的交易日
            detail_sql = f"""
            SELECT TradingDate, SecuMarket, IfTradingDay, IfWeekEnd, IfMonthEnd
            FROM QT_TradingDayNew
            WHERE TradingDate BETWEEN '{start_date}' AND '{end_date}'
                AND SecuMarket IN (83, 90)
                AND IfTradingDay = 1
            ORDER BY TradingDate
            """
            detail_df = db.read_sql(detail_sql)
            print(f"  具体交易日: {len(detail_df)} 天")
            for _, row in detail_df.iterrows():
                market_name = "深交所" if row['SecuMarket'] == 83 else "上交所"
                week_end = "周末" if row['IfWeekEnd'] == 1 else ""
                month_end = "月末" if row['IfMonthEnd'] == 1 else ""
                flags = f"({week_end}{month_end})".strip("()") if week_end or month_end else ""
                print(f"    {row['TradingDate']} {market_name} {flags}")
        
        # 3. 测试新股上市场景
        print("\n3. 测试新股上市场景:")
        test_new_stocks = [
            ('2024-01-15', '2024-01-19', '新股上市第1-5个交易日'),
            ('2024-01-15', '2024-01-26', '新股上市第1-10个交易日'),
        ]
        
        for listed_date, end_date, description in test_new_stocks:
            print(f"\n{description} (上市日期: {listed_date}):")
            
            # 计算交易日
            sql = f"""
            SELECT COUNT(*) as trading_days
            FROM QT_TradingDayNew
            WHERE TradingDate BETWEEN '{listed_date}' AND '{end_date}'
                AND SecuMarket IN (83, 90)
                AND IfTradingDay = 1
            """
            result = db.read_sql(sql)
            trading_days = result.iloc[0]['trading_days'] if not result.empty else 0
            
            print(f"  上市后交易日数: {trading_days}")
            
            if trading_days <= 5:
                print(f"  ✓ 属于新股前5个交易日")
            else:
                print(f"  ✗ 不属于新股前5个交易日")
        
        # 4. 检查2024年节假日数据
        print("\n4. 检查2024年中国股市交易日数据:")
        holiday_sql = """
        SELECT TradingDate, SecuMarket, IfTradingDay, IfWeekEnd, IfMonthEnd, IfQuarterEnd, IfYearEnd
        FROM QT_TradingDayNew
        WHERE TradingDate BETWEEN '2024-01-01' AND '2024-12-31'
            AND SecuMarket IN (83, 90)
        ORDER BY TradingDate
        """
        holiday_df = db.read_sql(holiday_sql)
        
        # 统计交易日和非交易日
        trading_days = holiday_df[holiday_df['IfTradingDay'] == 1]
        non_trading_days = holiday_df[holiday_df['IfTradingDay'] == 0]
        
        print(f"  总记录数: {len(holiday_df)}")
        print(f"  交易日数: {len(trading_days)}")
        print(f"  非交易日数: {len(non_trading_days)}")
        
        # 检查周末交易日（调休工作日）
        weekend_trading = trading_days[trading_days['IfWeekEnd'] == 1]
        if not weekend_trading.empty:
            print(f"  调休工作日: {len(weekend_trading)} 天")
            for _, row in weekend_trading.head(5).iterrows():
                market_name = "深交所" if row['SecuMarket'] == 83 else "上交所"
                print(f"    {row['TradingDate']} {market_name}")
        
        # 检查月末交易日
        month_end_trading = trading_days[trading_days['IfMonthEnd'] == 1]
        print(f"  月末交易日: {len(month_end_trading)} 天")
        
        # 检查年末交易日
        year_end_trading = trading_days[trading_days['IfYearEnd'] == 1]
        print(f"  年末交易日: {len(year_end_trading)} 天")
        
        db.close()
        print("\n" + "="*60)
        print("测试完成")
        print("="*60)
        
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == '__main__':
    test_trading_days_calculation() 