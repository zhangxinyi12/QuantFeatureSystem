#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
停牌原因分析汇总工具
分析停牌复牌表的各种原因分布和统计信息
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connector import JuyuanDB
import pandas as pd
from datetime import datetime, timedelta
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SuspendReasonAnalyzer:
    """停牌原因分析器"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        
        # 信息来源映射（根据CT_SystemConst表）
        self.info_source_map = {
            1: "交易所公告",
            2: "公司公告", 
            3: "证监会公告",
            4: "其他"
        }
        
        # 停牌事项说明映射
        self.suspend_statement_map = {
            1: "重大事项",
            2: "重大资产重组",
            3: "重大投资",
            4: "重大合同",
            5: "重大诉讼",
            6: "重大担保",
            7: "重大关联交易",
            8: "重大债务",
            9: "重大亏损",
            10: "其他"
        }
        
        # 停牌期限类型映射
        self.suspend_type_map = {
            1: "临时停牌",
            2: "长期停牌",
            3: "无限期停牌",
            4: "其他"
        }
    
    def get_suspend_data(self, start_date=None, end_date=None, exchanges=None):
        """获取停牌数据"""
        if start_date is None:
            start_date = "2024-01-01"
        if end_date is None:
            end_date = "2024-12-31"
        
        # 构建查询条件
        exchange_condition = ""
        if exchanges:
            exchange_list = "', '".join(exchanges)
            exchange_condition = f"AND s.Exchange IN ('{exchange_list}')"
        
        query = f"""
        SELECT 
            t.*,
            s.SecuCode, s.SecuAbbr, s.Exchange, s.ListDate, s.DelistDate
        FROM LC_SuspendResumption t
        INNER JOIN LC_SecuMain s ON t.SecuCode = s.SecuCode
        WHERE t.SuspendDate BETWEEN '{start_date}' AND '{end_date}'
        {exchange_condition}
        AND s.DelistDate IS NULL
        ORDER BY t.SuspendDate DESC
        """
        
        print(f"📊 查询停牌数据: {start_date} 到 {end_date}")
        df = self.db.read_sql(query)
        print(f"✅ 获取到 {len(df)} 条停牌记录")
        
        return df
    
    def analyze_suspend_reasons(self, df):
        """分析停牌原因分布"""
        print("\n=== 停牌原因分析 ===")
        
        # 基本统计
        total_records = len(df)
        print(f"📊 总停牌记录数: {total_records}")
        
        # 1. 停牌原因分析
        print(f"\n1. 停牌原因分布:")
        reason_counts = df['SuspendReason'].value_counts()
        for reason, count in reason_counts.head(10).items():
            percentage = (count / total_records) * 100
            print(f"   {reason}: {count} 次 ({percentage:.2f}%)")
        
        # 2. 信息来源分析
        print(f"\n2. 信息来源分布:")
        source_counts = df['InfoSource'].value_counts()
        for source_id, count in source_counts.items():
            source_name = self.info_source_map.get(source_id, f"未知({source_id})")
            percentage = (count / total_records) * 100
            print(f"   {source_name}: {count} 次 ({percentage:.2f}%)")
        
        # 3. 停牌事项说明分析
        print(f"\n3. 停牌事项说明分布:")
        statement_counts = df['SuspendStatement'].value_counts()
        for statement_id, count in statement_counts.head(10).items():
            statement_name = self.suspend_statement_map.get(statement_id, f"未知({statement_id})")
            percentage = (count / total_records) * 100
            print(f"   {statement_name}: {count} 次 ({percentage:.2f}%)")
        
        # 4. 停牌期限类型分析
        print(f"\n4. 停牌期限类型分布:")
        type_counts = df['SuspendType'].value_counts()
        for type_id, count in type_counts.items():
            type_name = self.suspend_type_map.get(type_id, f"未知({type_id})")
            percentage = (count / total_records) * 100
            print(f"   {type_name}: {count} 次 ({percentage:.2f}%)")
        
        # 5. 交易所分布
        print(f"\n5. 交易所分布:")
        exchange_counts = df['Exchange'].value_counts()
        for exchange, count in exchange_counts.items():
            percentage = (count / total_records) * 100
            print(f"   {exchange}: {count} 次 ({percentage:.2f}%)")
        
        return {
            'reason_counts': reason_counts,
            'source_counts': source_counts,
            'statement_counts': statement_counts,
            'type_counts': type_counts,
            'exchange_counts': exchange_counts
        }
    
    def analyze_suspend_duration(self, df):
        """分析停牌时长"""
        print(f"\n=== 停牌时长分析 ===")
        
        # 计算停牌时长
        df['SuspendDate'] = pd.to_datetime(df['SuspendDate'])
        df['ResumptionDate'] = pd.to_datetime(df['ResumptionDate'])
        
        # 只分析已复牌的记录
        resumed_df = df[df['ResumptionDate'].notna()]
        print(f"📊 已复牌记录数: {len(resumed_df)}")
        
        if len(resumed_df) == 0:
            print("❌ 没有已复牌的记录")
            return None
        
        # 计算停牌天数
        resumed_df['suspend_days'] = (resumed_df['ResumptionDate'] - resumed_df['SuspendDate']).dt.days
        
        # 基本统计
        print(f"📈 停牌时长统计:")
        print(f"   平均停牌天数: {resumed_df['suspend_days'].mean():.1f} 天")
        print(f"   中位数停牌天数: {resumed_df['suspend_days'].median():.1f} 天")
        print(f"   最长停牌天数: {resumed_df['suspend_days'].max()} 天")
        print(f"   最短停牌天数: {resumed_df['suspend_days'].min()} 天")
        
        # 按停牌原因分析时长
        print(f"\n📊 各停牌原因的平均时长:")
        reason_duration = resumed_df.groupby('SuspendReason')['suspend_days'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        for reason, row in reason_duration.head(10).iterrows():
            print(f"   {reason}: {row['mean']:.1f} 天 ({row['count']} 次)")
        
        # 按停牌类型分析时长
        print(f"\n📊 各停牌类型的平均时长:")
        type_duration = resumed_df.groupby('SuspendType')['suspend_days'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        for type_id, row in type_duration.iterrows():
            type_name = self.suspend_type_map.get(type_id, f"未知({type_id})")
            print(f"   {type_name}: {row['mean']:.1f} 天 ({row['count']} 次)")
        
        return resumed_df
    
    def analyze_monthly_trends(self, df):
        """分析月度趋势"""
        print(f"\n=== 月度趋势分析 ===")
        
        df['SuspendDate'] = pd.to_datetime(df['SuspendDate'])
        df['month'] = df['SuspendDate'].dt.to_period('M')
        
        # 月度停牌数量
        monthly_counts = df['month'].value_counts().sort_index()
        
        print(f"📊 月度停牌数量:")
        for month, count in monthly_counts.items():
            print(f"   {month}: {count} 次")
        
        # 月度主要停牌原因
        print(f"\n📊 月度主要停牌原因:")
        for month in monthly_counts.index:
            month_data = df[df['month'] == month]
            if len(month_data) > 0:
                top_reason = month_data['SuspendReason'].mode().iloc[0] if len(month_data['SuspendReason'].mode()) > 0 else "无"
                print(f"   {month}: {top_reason} ({len(month_data)} 次)")
        
        return monthly_counts
    
    def generate_reports(self, df, output_dir="../output/suspend_analysis"):
        """生成分析报告"""
        print(f"\n=== 生成分析报告 ===")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 生成详细CSV报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(output_dir, f"suspend_analysis_{timestamp}.csv")
        
        # 添加映射字段
        df_report = df.copy()
        df_report['InfoSourceName'] = df_report['InfoSource'].map(self.info_source_map)
        df_report['SuspendStatementName'] = df_report['SuspendStatement'].map(self.suspend_statement_map)
        df_report['SuspendTypeName'] = df_report['SuspendType'].map(self.suspend_type_map)
        
        df_report.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✅ 详细报告已保存: {csv_file}")
        
        # 2. 生成统计汇总
        summary_file = os.path.join(output_dir, f"suspend_summary_{timestamp}.csv")
        
        # 停牌原因汇总
        reason_summary = df['SuspendReason'].value_counts().reset_index()
        reason_summary.columns = ['停牌原因', '次数']
        reason_summary['占比'] = (reason_summary['次数'] / len(df) * 100).round(2)
        
        reason_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"✅ 统计汇总已保存: {summary_file}")
        
        # 3. 生成图表
        self.generate_charts(df, output_dir, timestamp)
        
        return csv_file, summary_file
    
    def generate_charts(self, df, output_dir, timestamp):
        """生成分析图表"""
        print(f"📊 生成分析图表...")
        
        # 设置图表样式
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('停牌原因分析报告', fontsize=16, fontweight='bold')
        
        # 1. 停牌原因分布饼图
        reason_counts = df['SuspendReason'].value_counts().head(8)
        axes[0, 0].pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('停牌原因分布 (Top 8)')
        
        # 2. 信息来源分布
        source_counts = df['InfoSource'].value_counts()
        source_names = [self.info_source_map.get(x, f"未知({x})") for x in source_counts.index]
        axes[0, 1].bar(source_names, source_counts.values)
        axes[0, 1].set_title('信息来源分布')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 停牌类型分布
        type_counts = df['SuspendType'].value_counts()
        type_names = [self.suspend_type_map.get(x, f"未知({x})") for x in type_counts.index]
        axes[1, 0].bar(type_names, type_counts.values)
        axes[1, 0].set_title('停牌类型分布')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 月度趋势
        df['SuspendDate'] = pd.to_datetime(df['SuspendDate'])
        monthly_counts = df['SuspendDate'].dt.to_period('M').value_counts().sort_index()
        axes[1, 1].plot(range(len(monthly_counts)), monthly_counts.values, marker='o')
        axes[1, 1].set_title('月度停牌数量趋势')
        axes[1, 1].set_xlabel('月份')
        axes[1, 1].set_ylabel('停牌数量')
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = os.path.join(output_dir, f"suspend_charts_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"✅ 分析图表已保存: {chart_file}")
        
        plt.close()

def main():
    """主函数"""
    try:
        # 连接数据库
        print("🔌 连接数据库...")
        db = JuyuanDB(use_ssh_tunnel=False)
        
        # 创建分析器
        analyzer = SuspendReasonAnalyzer(db)
        
        # 分析参数
        START_DATE = "2024-01-01"
        END_DATE = "2024-12-31"
        EXCHANGES = ["SSE", "SZSE"]  # 上交所和深交所
        
        # 获取数据
        df = analyzer.get_suspend_data(START_DATE, END_DATE, EXCHANGES)
        
        if df.empty:
            print("❌ 没有找到停牌数据")
            return
        
        # 执行分析
        analysis_results = analyzer.analyze_suspend_reasons(df)
        duration_analysis = analyzer.analyze_suspend_duration(df)
        monthly_trends = analyzer.analyze_monthly_trends(df)
        
        # 生成报告
        csv_file, summary_file = analyzer.generate_reports(df)
        
        # 关闭数据库连接
        db.close()
        
        print(f"\n✅ 停牌原因分析完成！")
        print(f"📁 报告保存在: ../output/suspend_analysis/")
        print(f"📄 详细报告: {csv_file}")
        print(f"📊 统计汇总: {summary_file}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 