#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成交量衍生特征使用示例
展示如何使用成交量特征工程模块
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.feature_engine.volume_features import (
    calculate_volume_features,
    calculate_absolute_volume,
    calculate_volume_change,
    calculate_accumulation_distribution,
    calculate_mfi,
    detect_volume_price_divergence,
    calculate_vwap,
    calculate_order_flow_imbalance,
    calculate_turnover_rate,
    detect_block_trades
)

def load_sample_data():
    """加载示例数据（这里使用模拟数据）"""
    # 创建模拟的股票数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # 生成更真实的股票价格序列
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]
    
    for i in range(1, 100):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    # 生成OHLCV数据
    data = {
        'Open': prices,
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Volume': np.random.randint(1000000, 5000000, 100)
    }
    
    # 确保High >= Low
    for i in range(100):
        data['High'][i] = max(data['High'][i], data['Close'][i])
        data['Low'][i] = min(data['Low'][i], data['Close'][i])
    
    df = pd.DataFrame(data, index=dates)
    return df

def example_1_basic_volume_features():
    """示例1: 基础成交量特征"""
    print("=== 示例1: 基础成交量特征 ===")
    
    # 加载数据
    df = load_sample_data()
    print(f"数据形状: {df.shape}")
    
    # 计算基础成交量特征
    result = calculate_absolute_volume(df, windows=[5, 20, 50])
    
    print("新增特征:")
    for col in result.columns:
        if col not in df.columns:
            print(f"  - {col}")
    
    # 显示最新数据
    print(f"\n最新数据:")
    print(f"  当期成交量: {result['Volume_Current'].iloc[-1]:,.0f}")
    print(f"  5日平均成交量: {result['Volume_MA_5'].iloc[-1]:,.0f}")
    print(f"  20日平均成交量: {result['Volume_MA_20'].iloc[-1]:,.0f}")
    print()

def example_2_volume_change_features():
    """示例2: 成交量变化特征"""
    print("=== 示例2: 成交量变化特征 ===")
    
    df = load_sample_data()
    
    # 计算成交量变化特征
    result = calculate_volume_change(df, windows=[5, 10, 20])
    
    print("新增特征:")
    for col in result.columns:
        if col not in df.columns:
            print(f"  - {col}")
    
    # 分析成交量变化
    latest_volume_change = result['Volume_Change_Rate_5'].iloc[-1]
    latest_momentum = result['Volume_Momentum_10'].iloc[-1]
    
    print(f"\n成交量分析:")
    print(f"  5日成交量变化率: {latest_volume_change:.2%}")
    print(f"  10日成交量动量: {latest_momentum:,.0f}")
    
    if latest_volume_change > 0.1:
        print("  → 成交量显著增加")
    elif latest_volume_change < -0.1:
        print("  → 成交量显著减少")
    else:
        print("  → 成交量相对稳定")
    print()

def example_3_volume_price_features():
    """示例3: 量价结合特征"""
    print("=== 示例3: 量价结合特征 ===")
    
    df = load_sample_data()
    
    # 计算A/D线
    result = calculate_accumulation_distribution(df)
    
    # 计算MFI（如果TA-Lib可用）
    try:
        result = calculate_mfi(result, period=14)
        mfi_available = True
    except ImportError:
        print("TA-Lib未安装，跳过MFI计算")
        mfi_available = False
    
    # 检测量价背离
    result = detect_volume_price_divergence(result, window=20)
    
    # 计算VWAP
    result = calculate_vwap(result, window=30)
    
    print("新增特征:")
    for col in result.columns:
        if col not in df.columns:
            print(f"  - {col}")
    
    # 分析结果
    print(f"\n量价分析:")
    print(f"  A/D线值: {result['AD_Line'].iloc[-1]:,.0f}")
    if mfi_available:
        mfi_value = result['MFI'].iloc[-1]
        print(f"  MFI值: {mfi_value:.2f}")
        if mfi_value > 80:
            print("  → 超买区域")
        elif mfi_value < 20:
            print("  → 超卖区域")
        else:
            print("  → 正常区域")
    
    print(f"  30日VWAP: {result['VWAP_30'].iloc[-1]:.2f}")
    print(f"  收盘价: {result['Close'].iloc[-1]:.2f}")
    
    # 检查量价背离
    divergence_cols = [col for col in result.columns if 'Divergence' in col]
    for col in divergence_cols:
        count = result[col].sum()
        if count > 0:
            print(f"  {col}: 检测到{count}次")
    print()

def example_4_turnover_rate():
    """示例4: 换手率特征"""
    print("=== 示例4: 换手率特征 ===")
    
    df = load_sample_data()
    
    # 假设流通股本为1亿股
    float_shares = 100000000
    
    result = calculate_turnover_rate(df, float_shares=float_shares)
    
    print("新增特征:")
    for col in result.columns:
        if col not in df.columns:
            print(f"  - {col}")
    
    # 分析换手率
    turnover_rate = result['Turnover_Rate'].iloc[-1]
    avg_turnover = result['Turnover_Rate'].mean()
    
    print(f"\n换手率分析:")
    print(f"  当日换手率: {turnover_rate:.2%}")
    print(f"  平均换手率: {avg_turnover:.2%}")
    
    if turnover_rate > avg_turnover * 2:
        print("  → 换手率异常活跃")
    elif turnover_rate < avg_turnover * 0.5:
        print("  → 换手率相对低迷")
    else:
        print("  → 换手率正常")
    print()

def example_5_comprehensive_features():
    """示例5: 综合成交量特征"""
    print("=== 示例5: 综合成交量特征 ===")
    
    df = load_sample_data()
    
    # 假设流通股本为1亿股
    float_shares = 100000000
    
    # 计算所有成交量特征
    result = calculate_volume_features(
        df, 
        float_shares=float_shares,
        vwap_windows=[15, 30, 60],
        volume_ma_windows=[5, 20, 50],
        volume_change_windows=[5, 10, 20]
    )
    
    print(f"原始特征数: {len(df.columns)}")
    print(f"处理后特征数: {len(result.columns)}")
    print(f"新增特征数: {len(result.columns) - len(df.columns)}")
    
    print("\n所有新增特征:")
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    new_features = [col for col in result.columns if col not in original_cols]
    
    # 按类别分组显示
    feature_categories = {
        '绝对成交量': [f for f in new_features if 'Volume_MA_' in f or 'Volume_Current' in f],
        '成交量变化': [f for f in new_features if 'Volume_Change' in f or 'Volume_Momentum' in f],
        '量价结合': [f for f in new_features if 'AD_Line' in f or 'MFI' in f or 'VWAP' in f or 'Divergence' in f],
        '换手率': [f for f in new_features if 'Turnover' in f]
    }
    
    for category, features in feature_categories.items():
        if features:
            print(f"\n{category}特征:")
            for feature in features:
                print(f"  - {feature}")
    
    print(f"\n总计: {len(new_features)}个特征")
    print()

def example_6_level2_features():
    """示例6: Level 2数据特征（模拟数据）"""
    print("=== 示例6: Level 2数据特征 ===")
    
    # 创建模拟的Level 2数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    
    base_price = 100
    data = {
        'Bid_Price1': [base_price + np.random.normal(0, 0.1) for _ in range(50)],
        'Bid_Size1': np.random.randint(1000, 10000, 50),
        'Ask_Price1': [base_price + np.random.normal(0, 0.1) for _ in range(50)],
        'Ask_Size1': np.random.randint(1000, 10000, 50),
        'Volume': np.random.randint(1000000, 5000000, 50)
    }
    
    # 确保Ask > Bid
    for i in range(50):
        if data['Ask_Price1'][i] <= data['Bid_Price1'][i]:
            data['Ask_Price1'][i] = data['Bid_Price1'][i] + 0.01
    
    df = pd.DataFrame(data, index=dates)
    
    # 计算订单流不平衡特征
    result = calculate_order_flow_imbalance(df)
    
    print("新增特征:")
    for col in result.columns:
        if col not in df.columns:
            print(f"  - {col}")
    
    # 分析订单流
    latest_ofi = result['OFI'].iloc[-1]
    latest_pressure = result['Price_Pressure'].iloc[-1]
    
    print(f"\n订单流分析:")
    print(f"  订单流不平衡: {latest_ofi:.3f}")
    print(f"  价格压力: {latest_pressure:.3f}")
    
    if latest_ofi > 0.1:
        print("  → 买盘压力较大")
    elif latest_ofi < -0.1:
        print("  → 卖盘压力较大")
    else:
        print("  → 买卖压力相对平衡")
    print()

def main():
    """主函数"""
    print("成交量衍生特征使用示例\n")
    
    try:
        # 运行各个示例
        example_1_basic_volume_features()
        example_2_volume_change_features()
        example_3_volume_price_features()
        example_4_turnover_rate()
        example_5_comprehensive_features()
        example_6_level2_features()
        
        print("所有示例运行完成！")
        
    except Exception as e:
        print(f"运行示例时出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 