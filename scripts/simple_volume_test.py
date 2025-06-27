#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的成交量衍生特征测试脚本
测试基本功能，不依赖TA-Lib
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_sample_data(n_days=50):
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

def calculate_absolute_volume_simple(df, windows=[5, 20]):
    """简化的绝对成交量特征计算"""
    df = df.copy()
    
    # 当期成交量
    df['Volume_Current'] = df['Volume']
    
    # 不同窗口的平均成交量
    for window in windows:
        df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
    
    return df

def calculate_volume_change_simple(df, windows=[5, 10]):
    """简化的成交量变化特征计算"""
    df = df.copy()
    
    # 成交量变化率
    for window in windows:
        df[f'Volume_Change_Rate_{window}'] = df['Volume'].pct_change(window)
    
    # 成交量动量
    for window in windows:
        df[f'Volume_Momentum_{window}'] = df['Volume'] - df['Volume'].shift(window)
    
    return df

def calculate_accumulation_distribution_simple(df):
    """简化的累积/派发线计算"""
    df = df.copy()
    
    # 计算资金流乘数
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)  # 处理除零情况
    
    # 计算资金流
    ad = clv * df['Volume']
    
    # 累积资金流
    df['AD_Line'] = ad.cumsum()
    
    return df

def calculate_vwap_simple(df, window=30):
    """简化的VWAP计算"""
    df = df.copy()
    
    # 计算典型价格
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    
    # 滚动VWAP
    rolling_value = (typical_price * df['Volume']).rolling(window).sum()
    rolling_volume = df['Volume'].rolling(window).sum()
    df[f'VWAP_{window}'] = rolling_value / rolling_volume
    
    return df

def calculate_turnover_rate_simple(df, float_shares=100000000):
    """简化的换手率计算"""
    df = df.copy()
    df['Turnover_Rate'] = df['Volume'] / float_shares
    return df

def test_basic_features():
    """测试基本特征功能"""
    print("=== 成交量衍生特征基本测试 ===\n")
    
    # 创建示例数据
    df = create_sample_data(30)
    print(f"原始数据形状: {df.shape}")
    print(f"数据列: {list(df.columns)}")
    print(f"数据范围: {df.index[0]} 到 {df.index[-1]}")
    print()
    
    # 测试绝对成交量特征
    print("1. 测试绝对成交量特征:")
    result = calculate_absolute_volume_simple(df, windows=[5, 20])
    print(f"   - 当期成交量: {result['Volume_Current'].iloc[-1]:,.0f}")
    print(f"   - 5日平均成交量: {result['Volume_MA_5'].iloc[-1]:,.0f}")
    print(f"   - 20日平均成交量: {result['Volume_MA_20'].iloc[-1]:,.0f}")
    print()
    
    # 测试成交量变化特征
    print("2. 测试成交量变化特征:")
    result = calculate_volume_change_simple(result, windows=[5, 10])
    print(f"   - 5日成交量变化率: {result['Volume_Change_Rate_5'].iloc[-1]:.2%}")
    print(f"   - 10日成交量动量: {result['Volume_Momentum_10'].iloc[-1]:,.0f}")
    print()
    
    # 测试A/D线
    print("3. 测试累积/派发线:")
    result = calculate_accumulation_distribution_simple(result)
    print(f"   - A/D线值: {result['AD_Line'].iloc[-1]:,.0f}")
    print()
    
    # 测试VWAP
    print("4. 测试VWAP:")
    result = calculate_vwap_simple(result, window=30)
    print(f"   - 30日VWAP: {result['VWAP_30'].iloc[-1]:.2f}")
    print(f"   - 收盘价: {result['Close'].iloc[-1]:.2f}")
    print()
    
    # 测试换手率
    print("5. 测试换手率:")
    result = calculate_turnover_rate_simple(result, float_shares=100000000)
    print(f"   - 换手率: {result['Turnover_Rate'].iloc[-1]:.2%}")
    print()
    
    # 显示所有新增特征
    print("6. 所有新增特征列:")
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    new_features = [col for col in result.columns if col not in original_cols]
    for i, feature in enumerate(new_features, 1):
        print(f"   {i:2d}. {feature}")
    
    print(f"\n总计新增特征数: {len(new_features)}")
    print("\n=== 测试完成 ===")

def main():
    """主函数"""
    try:
        test_basic_features()
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 