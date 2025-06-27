#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成交量衍生特征测试脚本
测试所有成交量相关特征的计算功能
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.feature_engine.volume_features import (
    calculate_absolute_volume,
    calculate_volume_change,
    calculate_accumulation_distribution,
    calculate_mfi,
    detect_volume_price_divergence,
    calculate_vwap,
    calculate_order_flow_imbalance,
    calculate_turnover_rate,
    detect_block_trades,
    calculate_volume_features
)

def create_sample_data(n_days=100):
    """创建示例OHLCV数据"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # 创建更真实的股票数据
    base_price = 100
    returns = np.random.normal(0, 0.02, n_days)  # 2%的日波动率
    prices = [base_price]
    
    for i in range(1, n_days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    # 生成OHLC数据
    data = {
        'Open': prices,
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Volume': np.random.randint(1000000, 5000000, n_days)
    }
    
    # 确保High >= Low
    for i in range(n_days):
        data['High'][i] = max(data['High'][i], data['Close'][i])
        data['Low'][i] = min(data['Low'][i], data['Close'][i])
    
    df = pd.DataFrame(data, index=dates)
    return df

def create_level2_sample_data(n_days=100):
    """创建Level 2数据示例"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    base_price = 100
    data = {
        'Bid_Price1': [base_price + np.random.normal(0, 0.1) for _ in range(n_days)],
        'Bid_Size1': np.random.randint(1000, 10000, n_days),
        'Ask_Price1': [base_price + np.random.normal(0, 0.1) for _ in range(n_days)],
        'Ask_Size1': np.random.randint(1000, 10000, n_days),
        'Volume': np.random.randint(1000000, 5000000, n_days)
    }
    
    # 确保Ask > Bid
    for i in range(n_days):
        if data['Ask_Price1'][i] <= data['Bid_Price1'][i]:
            data['Ask_Price1'][i] = data['Bid_Price1'][i] + 0.01
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_absolute_volume_features():
    """测试绝对成交量特征"""
    print("=== 测试绝对成交量特征 ===")
    df = create_sample_data()
    
    # 计算绝对成交量特征
    result = calculate_absolute_volume(df, windows=[5, 20, 50])
    
    print(f"原始数据形状: {df.shape}")
    print(f"处理后数据形状: {result.shape}")
    print("新增特征列:")
    for col in result.columns:
        if col not in df.columns:
            print(f"  - {col}")
    
    print(f"当期成交量: {result['Volume_Current'].iloc[-1]:,.0f}")
    print(f"5日平均成交量: {result['Volume_MA_5'].iloc[-1]:,.0f}")
    print(f"20日平均成交量: {result['Volume_MA_20'].iloc[-1]:,.0f}")
    print()

def test_volume_change_features():
    """测试成交量变化特征"""
    print("=== 测试成交量变化特征 ===")
    df = create_sample_data()
    
    # 计算成交量变化特征
    result = calculate_volume_change(df, windows=[5, 10, 20])
    
    print("新增特征列:")
    for col in result.columns:
        if col not in df.columns:
            print(f"  - {col}")
    
    print(f"5日成交量变化率: {result['Volume_Change_Rate_5'].iloc[-1]:.2%}")
    print(f"10日成交量动量: {result['Volume_Momentum_10'].iloc[-1]:,.0f}")
    print()

def test_volume_price_features():
    """测试量价结合特征"""
    print("=== 测试量价结合特征 ===")
    df = create_sample_data()
    
    # 计算A/D线
    result = calculate_accumulation_distribution(df)
    print(f"A/D线值: {result['AD_Line'].iloc[-1]:,.0f}")
    
    # 计算MFI
    result = calculate_mfi(result, period=14)
    print(f"MFI值: {result['MFI'].iloc[-1]:.2f}")
    
    # 检测量价背离
    result = detect_volume_price_divergence(result, window=20)
    print("量价背离特征:")
    for col in result.columns:
        if 'Divergence' in col:
            count = result[col].sum()
            print(f"  - {col}: {count}次")
    
    # 计算VWAP
    result = calculate_vwap(result, window=30)
    print(f"30日VWAP: {result['VWAP_30'].iloc[-1]:.2f}")
    print()

def test_turnover_rate():
    """测试换手率特征"""
    print("=== 测试换手率特征 ===")
    df = create_sample_data()
    
    # 假设流通股本为1亿股
    float_shares = 100000000
    
    result = calculate_turnover_rate(df, float_shares=float_shares)
    print(f"换手率: {result['Turnover_Rate'].iloc[-1]:.2%}")
    print()

def test_order_flow_imbalance():
    """测试订单流不平衡特征"""
    print("=== 测试订单流不平衡特征 ===")
    df = create_level2_sample_data()
    
    result = calculate_order_flow_imbalance(df)
    print(f"中间价: {result['Mid_Price'].iloc[-1]:.2f}")
    print(f"订单流不平衡: {result['OFI'].iloc[-1]:.3f}")
    print(f"价格压力: {result['Price_Pressure'].iloc[-1]:.3f}")
    print()

def test_block_trades():
    """测试大单检测特征"""
    print("=== 测试大单检测特征 ===")
    df = create_sample_data()
    
    # 添加Price列用于异常价格检测
    df['Price'] = df['Close']
    
    result = detect_block_trades(df, volume_multiplier=3, price_deviation=0.02, window=20)
    
    print("大单和异常交易检测:")
    print(f"  - 大单次数: {result['Block_Trade'].sum()}")
    print(f"  - 异常价格交易次数: {result['Abnormal_Price_Trade'].sum()}")
    print(f"  - 异常成交量交易次数: {result['Abnormal_Volume_Trade'].sum()}")
    print()

def test_comprehensive_volume_features():
    """测试综合成交量特征"""
    print("=== 测试综合成交量特征 ===")
    df = create_sample_data()
    
    # 假设流通股本为1亿股
    float_shares = 100000000
    
    result = calculate_volume_features(
        df, 
        float_shares=float_shares,
        vwap_windows=[15, 30],
        volume_ma_windows=[5, 20],
        volume_change_windows=[5, 10]
    )
    
    print(f"原始特征数: {len(df.columns)}")
    print(f"处理后特征数: {len(result.columns)}")
    print(f"新增特征数: {len(result.columns) - len(df.columns)}")
    
    print("\n所有新增特征:")
    for col in result.columns:
        if col not in df.columns:
            print(f"  - {col}")
    print()

def visualize_features():
    """可视化部分特征"""
    print("=== 生成特征可视化图表 ===")
    df = create_sample_data(200)
    float_shares = 100000000
    
    # 计算综合特征
    result = calculate_volume_features(df, float_shares=float_shares)
    
    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # 1. 价格与成交量
    ax1 = axes[0]
    ax1.plot(result.index, result['Close'], label='收盘价', color='blue', linewidth=1)
    ax1.set_ylabel('价格', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    
    ax1b = ax1.twinx()
    ax1b.bar(result.index, result['Volume'], alpha=0.3, color='gray', label='成交量')
    ax1b.set_ylabel('成交量', color='gray')
    ax1b.tick_params(axis='y', labelcolor='gray')
    ax1b.legend(loc='upper right')
    ax1.set_title('价格与成交量')
    
    # 2. A/D线与价格
    ax2 = axes[1]
    ax2.plot(result.index, result['Close'], label='收盘价', color='blue', linewidth=1)
    ax2.set_ylabel('价格', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper left')
    
    ax2b = ax2.twinx()
    ax2b.plot(result.index, result['AD_Line'], label='A/D线', color='green', linewidth=1)
    ax2b.set_ylabel('A/D线', color='green')
    ax2b.tick_params(axis='y', labelcolor='green')
    ax2b.legend(loc='upper right')
    ax2.set_title('价格与累积/派发线')
    
    # 3. MFI指标
    ax3 = axes[2]
    ax3.plot(result.index, result['Close'], label='收盘价', color='blue', linewidth=1)
    ax3.set_ylabel('价格', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.legend(loc='upper left')
    
    ax3b = ax3.twinx()
    ax3b.plot(result.index, result['MFI'], label='MFI', color='purple', linewidth=2)
    ax3b.axhline(80, color='red', linestyle='--', alpha=0.5, label='超买线')
    ax3b.axhline(20, color='green', linestyle='--', alpha=0.5, label='超卖线')
    ax3b.set_ylim(0, 100)
    ax3b.set_ylabel('MFI', color='purple')
    ax3b.tick_params(axis='y', labelcolor='purple')
    ax3b.legend(loc='upper right')
    ax3.set_title('价格与资金流量指标(MFI)')
    
    # 4. VWAP与价格
    ax4 = axes[3]
    ax4.plot(result.index, result['Close'], label='收盘价', color='blue', linewidth=1)
    ax4.plot(result.index, result['VWAP_30'], label='30日VWAP', color='orange', linewidth=2)
    ax4.set_ylabel('价格')
    ax4.legend()
    ax4.set_title('价格与成交量加权平均价(VWAP)')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'reports')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'volume_features_visualization.png'), 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {os.path.join(output_dir, 'volume_features_visualization.png')}")
    
    # 显示图表
    plt.show()

def main():
    """主测试函数"""
    print("成交量衍生特征测试开始...\n")
    
    try:
        # 测试各个特征模块
        test_absolute_volume_features()
        test_volume_change_features()
        test_volume_price_features()
        test_turnover_rate()
        test_order_flow_imbalance()
        test_block_trades()
        test_comprehensive_volume_features()
        
        # 生成可视化图表
        visualize_features()
        
        print("所有测试完成！成交量衍生特征功能正常。")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 