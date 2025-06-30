#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试开盘价比率分析的采样功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_open_close_ratio import OpenCloseRatioAnalyzer

def test_sampling_analysis():
    """测试采样分析功能"""
    print("开始测试开盘价比率分析的采样功能...")
    
    # 创建分析器（使用较短的时间范围进行测试）
    analyzer = OpenCloseRatioAnalyzer(
        start_date='2024-01-01',
        end_date='2024-03-31'  # 只测试前3个月
    )
    
    # 测试采样数据获取
    print("\n1. 测试采样数据获取...")
    df = analyzer.fetch_data_with_sampling()
    
    if df.empty:
        print("❌ 采样数据获取失败")
        return False
    
    print(f"✅ 采样数据获取成功，共 {len(df)} 行")
    print(f"   采样时间范围: {df['SampleYear'].min()}-{df['SampleMonth'].min():02d} 到 {df['SampleYear'].max()}-{df['SampleMonth'].max():02d}")
    print(f"   采样股票数: {df['SecuCode'].nunique()}")
    print(f"   采样板块数: {df['ListedSector'].nunique()}")
    
    # 测试比率计算
    print("\n2. 测试比率计算...")
    df_with_ratios = analyzer.calculate_ratios(df)
    
    if df_with_ratios.empty:
        print("❌ 比率计算失败")
        return False
    
    print(f"✅ 比率计算成功，有效数据 {len(df_with_ratios)} 行")
    print(f"   平均比率: {df_with_ratios['OpenCloseRatio'].mean():.4f}")
    print(f"   比率标准差: {df_with_ratios['OpenCloseRatio'].std():.4f}")
    
    # 测试采样统计
    print("\n3. 测试采样统计...")
    sampling_stats, overall_stats = analyzer.analyze_sampling_statistics(df)
    
    if sampling_stats is not None:
        print("✅ 采样统计成功")
        print(f"   统计月份数: {len(sampling_stats)}")
        print(f"   总采样数: {overall_stats['总采样数']}")
        print(f"   平均每月采样数: {overall_stats['平均每月采样数']:.1f}")
    else:
        print("❌ 采样统计失败")
        return False
    
    # 测试波动率分析
    print("\n4. 测试波动率分析...")
    volatility_stats, error_distribution = analyzer.analyze_volatility_and_noise(df_with_ratios)
    
    if volatility_stats:
        print("✅ 波动率分析成功")
        print(f"   波动率: {volatility_stats['波动率(标准差)']:.6f}")
        print(f"   波动率百分比: {volatility_stats['波动率(%)']:.4f}%")
        print(f"   ±0.1%覆盖率: {volatility_stats['±0.1%覆盖率(%)']:.2f}%")
    else:
        print("❌ 波动率分析失败")
        return False
    
    print("\n✅ 所有测试通过！")
    print(f"📁 结果文件保存在: {analyzer.output_dir}")
    
    return True

def main():
    """主函数"""
    success = test_sampling_analysis()
    
    if success:
        print("\n🎉 采样分析功能测试成功！")
        print("现在可以使用以下命令运行完整分析：")
        print("python scripts/analyze_open_close_ratio.py --sampling")
        print("或")
        print("python scripts/analyze_open_close_ratio.py --no-sampling")
    else:
        print("\n❌ 采样分析功能测试失败！")
        sys.exit(1)

if __name__ == '__main__':
    main() 