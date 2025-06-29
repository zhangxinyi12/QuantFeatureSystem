#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版年度数据库表空值检查工具
只需要指定年份，自动处理12个月的数据
按月份分别写入文件，避免并发写入问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connector import JuyuanDB
import pandas as pd
from datetime import datetime, timedelta
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import glob

def generate_yearly_monthly_ranges(year):
    """根据年份生成12个月的时间范围"""
    monthly_ranges = []
    
    for month in range(1, 13):
        # 计算当月开始日期
        start_date = datetime(year, month, 1)
        
        # 计算当月结束日期
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        monthly_ranges.append({
            'month': start_date.strftime("%Y-%m"),
            'start_date': start_date.strftime("%Y-%m-%d"),
            'end_date': end_date.strftime("%Y-%m-%d")
        })
    
    return monthly_ranges

def check_month_nulls(table_name, date_column, month_range, lock, output_dir, year):
    """检查单个月份的空值数据并直接写入文件"""
    month = month_range['month']
    start_date = month_range['start_date']
    end_date = month_range['end_date']
    
    try:
        # 为每个线程创建独立的数据库连接
        db = JuyuanDB(use_ssh_tunnel=False)
        
        # 查询数据
        query = f"""
        SELECT * FROM {table_name} 
        WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
        """
        
        df = db.read_sql(query)
        db.close()
        
        if df.empty:
            with lock:
                print(f"📅 {month}: 无数据")
            return {
                'month': month,
                'start_date': start_date,
                'end_date': end_date,
                'total_records': 0,
                'null_records': [],
                'status': 'no_data',
                'output_file': None
            }
        
        # 检查空值
        null_records = []
        for index, row in df.iterrows():
            null_columns = []
            for column in df.columns:
                if pd.isna(row[column]) or row[column] is None:
                    null_columns.append(column)
            
            if null_columns:  # 如果这一行有空值
                record = {
                    'month': month,
                    'row_index': index,
                    'has_nulls': '是',
                    'null_columns': ', '.join(null_columns),
                    'null_count': len(null_columns)
                }
                
                # 添加该行的所有原始数据
                for column in df.columns:
                    record[column] = row[column]
                
                null_records.append(record)
        
        # 直接写入月度文件
        output_file = None
        if null_records:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成月度文件名 - 格式: year_month
            output_file = os.path.join(output_dir, f"{year}_{month}.csv")
            
            # 写入CSV文件
            fieldnames = list(null_records[0].keys())
            with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(null_records)
        
        with lock:
            print(f"📅 {month}: {len(df)} 条记录, {len(null_records)} 条空值")
            if output_file:
                print(f"   💾 已保存到: {output_file}")
        
        return {
            'month': month,
            'start_date': start_date,
            'end_date': end_date,
            'total_records': len(df),
            'null_records': null_records,
            'status': 'success',
            'output_file': output_file
        }
        
    except Exception as e:
        with lock:
            print(f"❌ {month}: 检查失败 - {e}")
        return {
            'month': month,
            'start_date': start_date,
            'end_date': end_date,
            'total_records': 0,
            'null_records': [],
            'status': 'error',
            'error': str(e),
            'output_file': None
        }

def merge_monthly_files(output_dir, table_name, year):
    """合并所有月度文件为一个年度文件"""
    # 查找所有月度文件 - 格式: year_month.csv
    pattern = os.path.join(output_dir, f"{year}_*.csv")
    monthly_files = glob.glob(pattern)
    
    if not monthly_files:
        print("📁 没有找到月度文件需要合并")
        return None
    
    # 合并所有数据
    all_records = []
    for file_path in sorted(monthly_files):
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                records = list(reader)
                all_records.extend(records)
                print(f"📄 读取文件: {os.path.basename(file_path)} ({len(records)} 条记录)")
        except Exception as e:
            print(f"❌ 读取文件失败 {file_path}: {e}")
    
    if not all_records:
        print("📁 没有数据需要合并")
        return None
    
    # 写入合并文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = os.path.join(output_dir, f"yearly_null_check_{table_name}_{year}_{timestamp}.csv")
    
    fieldnames = list(all_records[0].keys())
    with open(merged_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    
    print(f"✅ 年度合并文件已保存: {merged_file}")
    print(f"📊 总计 {len(all_records)} 条记录")
    
    return merged_file

def show_statistics_from_files(output_dir, table_name, year):
    """从文件中读取并显示统计信息"""
    # 查找所有月度文件 - 格式: year_month.csv
    pattern = os.path.join(output_dir, f"{year}_*.csv")
    monthly_files = glob.glob(pattern)
    
    if not monthly_files:
        print("📁 没有找到月度文件")
        return
    
    print(f"\n=== 统计信息 ===")
    
    # 按月份统计
    monthly_stats = {}
    field_stats = {}
    all_records = []
    
    for file_path in sorted(monthly_files):
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                records = list(reader)
                all_records.extend(records)
                
                # 统计月度数据
                for record in records:
                    month = record['month']
                    if month not in monthly_stats:
                        monthly_stats[month] = 0
                    monthly_stats[month] += 1
                    
                    # 统计字段数据
                    null_columns = record['null_columns'].split(', ')
                    for column in null_columns:
                        if column not in field_stats:
                            field_stats[column] = 0
                        field_stats[column] += 1
                
                print(f"📄 {os.path.basename(file_path)}: {len(records)} 条记录")
                
        except Exception as e:
            print(f"❌ 读取文件失败 {file_path}: {e}")
    
    # 显示月度统计
    print(f"\n各月份空值记录统计:")
    for month in sorted(monthly_stats.keys()):
        count = monthly_stats[month]
        print(f"  {month}: {count} 条空值记录")
    
    # 显示字段统计
    print(f"\n各字段空值统计:")
    for field, count in sorted(field_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {field}: {count} 次")
    
    # 显示空值最多的记录
    if all_records:
        max_null_record = max(all_records, key=lambda x: int(x['null_count']))
        print(f"\n空值最多的记录:")
        print(f"  月份: {max_null_record['month']}")
        print(f"  行索引: {max_null_record['row_index']}")
        print(f"  空值字段数: {max_null_record['null_count']}")
        print(f"  空值字段: {max_null_record['null_columns']}")

def check_table_nulls_by_year(table_name, date_column, year, max_workers=12, output_dir=None):
    """根据年份检查表在整年的空值数据"""
    print(f"🔍 多线程检查表 {table_name} 的空值数据")
    print(f"📅 年份: {year}")
    print(f"🧵 使用 {max_workers} 个线程并行处理12个月")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = f"../output/null_check_{table_name}_{year}"
    
    print(f"📁 输出目录: {output_dir}")
    
    # 生成12个月的时间范围
    monthly_ranges = generate_yearly_monthly_ranges(year)
    print(f"📊 分为 {len(monthly_ranges)} 个月度任务")
    
    # 显示月度时间范围
    print("\n=== 月度时间范围 ===")
    for month_range in monthly_ranges:
        print(f"  {month_range['month']}: {month_range['start_date']} 到 {month_range['end_date']}")
    
    total_records = 0
    error_months = []
    lock = threading.Lock()
    
    start_time = time.time()
    
    # 使用线程池执行任务
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_month = {
            executor.submit(check_month_nulls, table_name, date_column, month_range, lock, output_dir, year): month_range
            for month_range in monthly_ranges
        }
        
        # 收集结果
        for future in as_completed(future_to_month):
            result = future.result()
            
            if result['status'] == 'success':
                total_records += result['total_records']
            elif result['status'] == 'error':
                error_months.append(result['month'])
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n=== 多线程检查完成 ===")
    print(f"⏱️  总耗时: {processing_time:.2f} 秒")
    print(f"📊 总记录数: {total_records}")
    
    if error_months:
        print(f"❌ 处理失败的月份: {', '.join(error_months)}")
    
    # 合并月度文件
    print(f"\n=== 合并月度文件 ===")
    merged_file = merge_monthly_files(output_dir, table_name, year)
    
    # 显示统计信息
    show_statistics_from_files(output_dir, table_name, year)
    
    if merged_file:
        print(f"\n✅ 年度检查完成！")
        print(f"📁 月度文件保存在: {output_dir}")
        print(f"📄 合并文件: {merged_file}")
    else:
        print(f"\n✅ 年度检查完成！没有发现空值数据")

if __name__ == "__main__":
    # 示例用法 - 修改这里的参数
    TABLE_NAME = "LC_SuspendResumption"  # 表名
    DATE_COLUMN = "SuspendDate"         # 日期字段名
    YEAR = 2024                          # 年份
    MAX_WORKERS = 12                     # 线程数量
    
    # 输出目录（可选，不指定则自动生成）
    OUTPUT_DIR = None  # 自动生成: ../output/null_check_LC_SuspendResumption_2024
    
    # 执行检查
    check_table_nulls_by_year(
        table_name=TABLE_NAME,
        date_column=DATE_COLUMN,
        year=YEAR,
        max_workers=MAX_WORKERS,
        output_dir=OUTPUT_DIR
    )
    
    # 检查其他表示例
    # check_table_nulls_by_year(
    #     table_name="LC_StockPrice",
    #     date_column="TradeDate",
    #     year=2024,
    #     output_dir="../output/stock_price_check_2024"
    # ) 