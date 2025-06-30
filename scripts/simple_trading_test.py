#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试交易日判断是否正确
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connector import JuyuanDB
import pandas as pd
from datetime import datetime, timedelta

def test_trading_days():
    """测试交易日判断"""
    print("="*50)
    print("测试交易日判断是否正确")
    print("="*50)
    
    try:
        db = JuyuanDB()
        
        # 测试1: 检查表结构
        print("\n1. 检查QT_TradingDayNew表结构:")
        try:
            sql = """
            SELECT TOP 3 *
            FROM QT_TradingDayNew
            WHERE SecuMarket IN (83, 90)
            ORDER BY TradingDate
            """
            df = db.read_sql(sql)
            print(f"表字段: {list(df.columns)}")
            print("前3行数据:")
            for _, row in df.iterrows():
                print(f"  {row['TradingDate']} - 市场{row['SecuMarket']} - 交易日{row['IfTradingDay']}")
        except Exception as e:
            print(f"检查表结构失败: {e}")
            return
        
        # 测试2: 测试几个具体的日期范围
        print("\n2. 测试具体日期范围的交易日计算:")
        test_cases = [
            ('2024-01-01', '2024-01-03', '新年假期'),
            ('2024-01-15', '2024-01-19', '正常工作日'),
            ('2024-02-10', '2024-02-16', '春节假期'),
        ]
        
        for start_date, end_date, description in test_cases:
            print(f"\n{description} ({start_date} 到 {end_date}):")
            
            # 计算交易日
            sql = f"""
            SELECT COUNT(*) as trading_days
            FROM QT_TradingDayNew
            WHERE TradingDate BETWEEN '{start_date}' AND '{end_date}'
                AND SecuMarket IN (83, 90)
                AND IfTradingDay = 1
            """
            result = db.read_sql(sql)
            trading_days = result.iloc[0]['trading_days'] if not result.empty else 0
            
            # 计算自然日
            natural_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
            
            print(f"  自然日数: {natural_days}")
            print(f"  交易日数: {trading_days}")
            print(f"  差异: {natural_days - trading_days}")
            
            # 显示具体交易日
            detail_sql = f"""
            SELECT TradingDate, SecuMarket, IfTradingDay
            FROM QT_TradingDayNew
            WHERE TradingDate BETWEEN '{start_date}' AND '{end_date}'
                AND SecuMarket IN (83, 90)
            ORDER BY TradingDate
            """
            detail_df = db.read_sql(detail_sql)
            print(f"  详细日期:")
            for _, row in detail_df.iterrows():
                market = "深交所" if row['SecuMarket'] == 83 else "上交所"
                trading = "交易日" if row['IfTradingDay'] == 1 else "非交易日"
                print(f"    {row['TradingDate']} {market} {trading}")
        
        # 测试3: 测试新股场景
        print("\n3. 测试新股上市场景:")
        test_new_stocks = [
            ('2024-01-15', '2024-01-15', '上市当天'),
            ('2024-01-15', '2024-01-19', '上市前5个交易日'),
            ('2024-01-15', '2024-01-26', '上市前10个交易日'),
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
        
        # 测试4: 检查2024年1月的交易日分布
        print("\n4. 检查2024年1月交易日分布:")
        month_sql = """
        SELECT TradingDate, SecuMarket, IfTradingDay, IfWeekEnd, IfMonthEnd
        FROM QT_TradingDayNew
        WHERE TradingDate BETWEEN '2024-01-01' AND '2024-01-31'
            AND SecuMarket IN (83, 90)
        ORDER BY TradingDate
        """
        month_df = db.read_sql(month_sql)
        
        trading_days = month_df[month_df['IfTradingDay'] == 1]
        non_trading_days = month_df[month_df['IfTradingDay'] == 0]
        
        print(f"  1月总记录数: {len(month_df)}")
        print(f"  交易日数: {len(trading_days)}")
        print(f"  非交易日数: {len(non_trading_days)}")
        
        print(f"  具体交易日:")
        for _, row in trading_days.iterrows():
            market = "深交所" if row['SecuMarket'] == 83 else "上交所"
            week_end = "周末" if row['IfWeekEnd'] == 1 else ""
            month_end = "月末" if row['IfMonthEnd'] == 1 else ""
            flags = f"({week_end}{month_end})".strip("()") if week_end or month_end else ""
            print(f"    {row['TradingDate']} {market} {flags}")
        
        db.close()
        print("\n" + "="*50)
        print("测试完成")
        print("="*50)
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == '__main__':
    test_trading_days() 