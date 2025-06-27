#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标特征测试脚本
测试所有技术指标的计算功能
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.feature_engine.technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_stochastic,
    calculate_williams_r,
    calculate_atr,
    detect_candlestick_patterns,
    calculate_seasonal_features,
    calculate_technical_features,
    get_technical_feature_summary
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

def test_rsi():
    """测试RSI指标"""
    print("=== 测试RSI指标 ===")
    df = create_sample_data()
    
    # 计算RSI
    rsi = calculate_rsi(df['Close'], period=14)
    
    print(f"RSI当前值: {rsi.iloc[-1]:.2f}")
    print(f"RSI平均值: {rsi.mean():.2f}")
    print(f"RSI标准差: {rsi.std():.2f}")
    print(f"超买次数 (>70): {(rsi > 70).sum()}")
    print(f"超卖次数 (<30): {(rsi < 30).sum()}")
    print()

def test_macd():
    """测试MACD指标"""
    print("=== 测试MACD指标 ===")
    df = create_sample_data()
    
    # 计算MACD
    macd_df = calculate_macd(df['Close'])
    
    print("MACD指标:")
    print(f"  当前MACD: {macd_df['MACD'].iloc[-1]:.4f}")
    print(f"  当前信号线: {macd_df['MACD_Signal'].iloc[-1]:.4f}")
    print(f"  当前柱状图: {macd_df['MACD_Histogram'].iloc[-1]:.4f}")
    
    # 检测金叉死叉
    macd = macd_df['MACD'].dropna()
    signal = macd_df['MACD_Signal'].dropna()
    
    golden_cross = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).sum()
    death_cross = ((macd < signal) & (macd.shift(1) >= signal.shift(1))).sum()
    
    print(f"  金叉次数: {golden_cross}")
    print(f"  死叉次数: {death_cross}")
    print()

def test_bollinger_bands():
    """测试布林带指标"""
    print("=== 测试布林带指标 ===")
    df = create_sample_data()
    
    # 计算布林带
    bb_df = calculate_bollinger_bands(df['Close'])
    
    print("布林带指标:")
    print(f"  当前上轨: {bb_df['BB_Upper'].iloc[-1]:.2f}")
    print(f"  当前中轨: {bb_df['BB_Middle'].iloc[-1]:.2f}")
    print(f"  当前下轨: {bb_df['BB_Lower'].iloc[-1]:.2f}")
    print(f"  当前位置: {bb_df['BB_Position'].iloc[-1]:.2f}")
    print(f"  当前宽度: {bb_df['BB_Width'].iloc[-1]:.2f}")
    
    # 检测突破
    bb_pos = bb_df['BB_Position'].dropna()
    upper_break = (bb_pos > 1).sum()
    lower_break = (bb_pos < 0).sum()
    
    print(f"  突破上轨次数: {upper_break}")
    print(f"  突破下轨次数: {lower_break}")
    print()

def test_stochastic():
    """测试随机指标"""
    print("=== 测试随机指标 ===")
    df = create_sample_data()
    
    # 计算随机指标
    stoch_df = calculate_stochastic(df['High'], df['Low'], df['Close'])
    
    print("随机指标:")
    print(f"  当前%K: {stoch_df['Stoch_K'].iloc[-1]:.2f}")
    print(f"  当前%D: {stoch_df['Stoch_D'].iloc[-1]:.2f}")
    
    # 检测超买超卖
    k = stoch_df['Stoch_K'].dropna()
    d = stoch_df['Stoch_D'].dropna()
    
    k_overbought = (k > 80).sum()
    k_oversold = (k < 20).sum()
    d_overbought = (d > 80).sum()
    d_oversold = (d < 20).sum()
    
    print(f"  %K超买次数: {k_overbought}")
    print(f"  %K超卖次数: {k_oversold}")
    print(f"  %D超买次数: {d_overbought}")
    print(f"  %D超卖次数: {d_oversold}")
    print()

def test_williams_r():
    """测试威廉指标"""
    print("=== 测试威廉指标 ===")
    df = create_sample_data()
    
    # 计算威廉指标
    williams_r = calculate_williams_r(df['High'], df['Low'], df['Close'])
    
    print("威廉指标:")
    print(f"  当前值: {williams_r.iloc[-1]:.2f}")
    print(f"  平均值: {williams_r.mean():.2f}")
    print(f"  超买次数 (<-80): {(williams_r < -80).sum()}")
    print(f"  超卖次数 (>-20): {(williams_r > -20).sum()}")
    print()

def test_atr():
    """测试ATR指标"""
    print("=== 测试ATR指标 ===")
    df = create_sample_data()
    
    # 计算ATR
    atr = calculate_atr(df['High'], df['Low'], df['Close'])
    
    print("ATR指标:")
    print(f"  当前值: {atr.iloc[-1]:.4f}")
    print(f"  平均值: {atr.mean():.4f}")
    print(f"  最大值: {atr.max():.4f}")
    print(f"  最小值: {atr.min():.4f}")
    print()

def test_candlestick_patterns():
    """测试K线形态检测"""
    print("=== 测试K线形态检测 ===")
    df = create_sample_data()
    
    # 检测K线形态
    result = detect_candlestick_patterns(df)
    
    pattern_cols = ['Hammer', 'Hanging_Man', 'Doji', 'Bullish_Engulfing', 'Bearish_Engulfing']
    
    print("K线形态统计:")
    for col in pattern_cols:
        if col in result.columns:
            count = result[col].sum()
            print(f"  {col}: {count}次")
    print()

def test_seasonal_features():
    """测试季节性特征"""
    print("=== 测试季节性特征 ===")
    df = create_sample_data()
    
    # 计算季节性特征
    result = calculate_seasonal_features(df)
    
    seasonal_cols = ['DayOfWeek', 'Month', 'Quarter', 'DayOfYear', 'WeekOfYear',
                    'IsMonthStart', 'IsMonthEnd', 'IsQuarterStart', 'IsQuarterEnd',
                    'IsYearStart', 'IsYearEnd']
    
    print("季节性特征:")
    for col in seasonal_cols:
        if col in result.columns:
            if col in ['DayOfWeek', 'Month', 'Quarter', 'DayOfYear', 'WeekOfYear']:
                print(f"  {col}: {result[col].iloc[-1]}")
            else:
                print(f"  {col}: {result[col].iloc[-1]}")
    print()

def test_comprehensive_technical_features():
    """测试综合技术指标特征"""
    print("=== 测试综合技术指标特征 ===")
    df = create_sample_data()
    
    # 计算所有技术指标
    result = calculate_technical_features(df)
    
    print(f"原始数据形状: {df.shape}")
    print(f"处理后数据形状: {result.shape}")
    
    # 统计新增特征
    new_features = [col for col in result.columns if col not in df.columns]
    print(f"新增特征数量: {len(new_features)}")
    
    # 按类型分组显示特征
    feature_groups = {
        '震荡指标': ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Stoch_K', 'Stoch_D', 'Williams_R'],
        '趋势指标': ['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position'],
        '波动率指标': ['ATR'],
        'K线形态': ['Hammer', 'Hanging_Man', 'Doji', 'Bullish_Engulfing', 'Bearish_Engulfing'],
        '季节性特征': ['DayOfWeek', 'Month', 'Quarter', 'DayOfYear', 'WeekOfYear']
    }
    
    for group_name, features in feature_groups.items():
        found_features = [f for f in features if f in new_features]
        if found_features:
            print(f"\n{group_name}:")
            for feature in found_features:
                print(f"  - {feature}")
    
    print()

def test_feature_summary():
    """测试特征摘要功能"""
    print("=== 测试特征摘要功能 ===")
    df = create_sample_data()
    
    # 计算技术指标
    result = calculate_technical_features(df)
    
    # 获取特征摘要
    summary = get_technical_feature_summary(result)
    
    print("特征摘要:")
    for category, details in summary.items():
        print(f"\n{category}:")
        for key, value in details.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    print()

def visualize_features():
    """可视化技术指标"""
    print("=== 可视化技术指标 ===")
    df = create_sample_data(200)
    
    # 计算技术指标
    result = calculate_technical_features(df)
    
    # 创建子图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('技术指标可视化', fontsize=16)
    
    # 价格和布林带
    axes[0, 0].plot(result.index, result['Close'], label='收盘价', alpha=0.7)
    axes[0, 0].plot(result.index, result['BB_Upper'], label='布林带上轨', alpha=0.7)
    axes[0, 0].plot(result.index, result['BB_Middle'], label='布林带中轨', alpha=0.7)
    axes[0, 0].plot(result.index, result['BB_Lower'], label='布林带下轨', alpha=0.7)
    axes[0, 0].set_title('价格和布林带')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RSI
    axes[0, 1].plot(result.index, result['RSI'], label='RSI', color='purple')
    axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买线')
    axes[0, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖线')
    axes[0, 1].set_title('RSI指标')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MACD
    axes[1, 0].plot(result.index, result['MACD'], label='MACD', color='blue')
    axes[1, 0].plot(result.index, result['MACD_Signal'], label='信号线', color='red')
    axes[1, 0].bar(result.index, result['MACD_Histogram'], label='柱状图', alpha=0.3, color='gray')
    axes[1, 0].set_title('MACD指标')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 随机指标
    axes[1, 1].plot(result.index, result['Stoch_K'], label='%K', color='blue')
    axes[1, 1].plot(result.index, result['Stoch_D'], label='%D', color='red')
    axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='超买线')
    axes[1, 1].axhline(y=20, color='g', linestyle='--', alpha=0.7, label='超卖线')
    axes[1, 1].set_title('随机指标')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 威廉指标
    axes[2, 0].plot(result.index, result['Williams_R'], label='Williams %R', color='orange')
    axes[2, 0].axhline(y=-20, color='r', linestyle='--', alpha=0.7, label='超买线')
    axes[2, 0].axhline(y=-80, color='g', linestyle='--', alpha=0.7, label='超卖线')
    axes[2, 0].set_title('威廉指标')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # ATR
    axes[2, 1].plot(result.index, result['ATR'], label='ATR', color='brown')
    axes[2, 1].set_title('ATR指标')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/reports/technical_indicators.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("技术指标图表已保存到 output/reports/technical_indicators.png")
    print()

def main():
    """主测试函数"""
    print("开始技术指标特征测试...\n")
    
    # 创建输出目录
    os.makedirs('output/reports', exist_ok=True)
    
    # 运行各项测试
    test_rsi()
    test_macd()
    test_bollinger_bands()
    test_stochastic()
    test_williams_r()
    test_atr()
    test_candlestick_patterns()
    test_seasonal_features()
    test_comprehensive_technical_features()
    test_feature_summary()
    
    # 可视化测试
    try:
        visualize_features()
    except Exception as e:
        print(f"可视化测试失败: {e}")
    
    print("技术指标特征测试完成！")

if __name__ == "__main__":
    main() 