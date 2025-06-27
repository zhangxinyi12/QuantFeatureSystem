#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成交量衍生特征演示脚本
展示如何使用成交量特征工程模块
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_demo_data():
    """创建演示数据"""
    print("创建演示数据...")
    
    # 生成模拟的股票数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    
    # 生成价格序列
    base_price = 100
    returns = np.random.normal(0, 0.02, 60)
    prices = [base_price]
    
    for i in range(1, 60):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    # 生成OHLCV数据
    data = {
        'Open': prices,
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Volume': np.random.randint(1000000, 5000000, 60)
    }
    
    # 确保High >= Low
    for i in range(60):
        data['High'][i] = max(data['High'][i], data['Close'][i])
        data['Low'][i] = min(data['Low'][i], data['Close'][i])
    
    df = pd.DataFrame(data, index=dates)
    print(f"数据形状: {df.shape}")
    print(f"数据范围: {df.index[0]} 到 {df.index[-1]}")
    return df

def demo_basic_volume_features(df):
    """演示基础成交量特征"""
    print("\n=== 演示基础成交量特征 ===")
    
    try:
        from src.processing.feature_engine.vol import calculate_absolute_volume, calculate_volume_change
        
        # 计算绝对成交量特征
        result = calculate_absolute_volume(df, windows=[5, 20])
        print("✓ 绝对成交量特征计算完成")
        print(f"  - 当期成交量: {result['Volume_Current'].iloc[-1]:,.0f}")
        print(f"  - 5日平均成交量: {result['Volume_MA_5'].iloc[-1]:,.0f}")
        print(f"  - 20日平均成交量: {result['Volume_MA_20'].iloc[-1]:,.0f}")
        
        # 计算成交量变化特征
        result = calculate_volume_change(result, windows=[5, 10])
        print("✓ 成交量变化特征计算完成")
        print(f"  - 5日成交量变化率: {result['Volume_Change_Rate_5'].iloc[-1]:.2%}")
        print(f"  - 10日成交量动量: {result['Volume_Momentum_10'].iloc[-1]:,.0f}")
        
        return result
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        return df

def demo_volume_price_features(df):
    """演示量价结合特征"""
    print("\n=== 演示量价结合特征 ===")
    
    try:
        from src.processing.feature_engine.vol import (
            calculate_accumulation_distribution,
            calculate_vwap,
            detect_volume_price_divergence
        )
        
        # 计算A/D线
        result = calculate_accumulation_distribution(df)
        print("✓ 累积/派发线计算完成")
        print(f"  - A/D线值: {result['AD_Line'].iloc[-1]:,.0f}")
        
        # 计算VWAP
        result = calculate_vwap(result, window=30)
        print("✓ VWAP计算完成")
        print(f"  - 30日VWAP: {result['VWAP_30'].iloc[-1]:.2f}")
        print(f"  - 收盘价: {result['Close'].iloc[-1]:.2f}")
        
        # 检测量价背离
        result = detect_volume_price_divergence(result, window=20)
        print("✓ 量价背离检测完成")
        
        # 检查背离情况
        divergence_cols = [col for col in result.columns if 'Divergence' in col]
        for col in divergence_cols:
            count = result[col].sum()
            if count > 0:
                print(f"  - {col}: 检测到{count}次")
        
        return result
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        return df

def demo_mfi_feature(df):
    """演示MFI特征（如果TA-Lib可用）"""
    print("\n=== 演示MFI特征 ===")
    
    try:
        from src.processing.feature_engine.vol import calculate_mfi
        
        result = calculate_mfi(df, period=14)
        mfi_value = result['MFI'].iloc[-1]
        
        print("✓ MFI计算完成")
        print(f"  - MFI值: {mfi_value:.2f}")
        
        if mfi_value > 80:
            print("  - 状态: 超买区域")
        elif mfi_value < 20:
            print("  - 状态: 超卖区域")
        else:
            print("  - 状态: 正常区域")
        
        return result
        
    except ImportError:
        print("✗ TA-Lib未安装，跳过MFI计算")
        return df
    except Exception as e:
        print(f"✗ MFI计算失败: {e}")
        return df

def demo_turnover_rate(df):
    """演示换手率特征"""
    print("\n=== 演示换手率特征 ===")
    
    try:
        from src.processing.feature_engine.vol import calculate_turnover_rate
        
        # 假设流通股本为1亿股
        float_shares = 100000000
        
        result = calculate_turnover_rate(df, float_shares=float_shares)
        turnover_rate = result['Turnover_Rate'].iloc[-1]
        avg_turnover = result['Turnover_Rate'].mean()
        
        print("✓ 换手率计算完成")
        print(f"  - 当日换手率: {turnover_rate:.2%}")
        print(f"  - 平均换手率: {avg_turnover:.2%}")
        
        if turnover_rate > avg_turnover * 2:
            print("  - 状态: 换手率异常活跃")
        elif turnover_rate < avg_turnover * 0.5:
            print("  - 状态: 换手率相对低迷")
        else:
            print("  - 状态: 换手率正常")
        
        return result
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        return df

def demo_comprehensive_features(df):
    """演示综合成交量特征"""
    print("\n=== 演示综合成交量特征 ===")
    
    try:
        from src.processing.feature_engine.vol import calculate_volume_features
        
        # 假设流通股本为1亿股
        float_shares = 100000000
        
        # 计算所有成交量特征
        result = calculate_volume_features(
            df, 
            float_shares=float_shares,
            vwap_windows=[15, 30],
            volume_ma_windows=[5, 20],
            volume_change_windows=[5, 10]
        )
        
        print("✓ 综合成交量特征计算完成")
        print(f"  - 原始特征数: {len(df.columns)}")
        print(f"  - 处理后特征数: {len(result.columns)}")
        print(f"  - 新增特征数: {len(result.columns) - len(df.columns)}")
        
        # 显示新增特征
        original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        new_features = [col for col in result.columns if col not in original_cols]
        
        print("\n新增特征类别:")
        categories = {
            '绝对成交量': [f for f in new_features if 'Volume_MA_' in f or 'Volume_Current' in f],
            '成交量变化': [f for f in new_features if 'Volume_Change' in f or 'Volume_Momentum' in f],
            '量价结合': [f for f in new_features if 'AD_Line' in f or 'VWAP' in f or 'Divergence' in f],
            '换手率': [f for f in new_features if 'Turnover' in f]
        }
        
        for category, features in categories.items():
            if features:
                print(f"  {category}: {len(features)}个特征")
        
        return result
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        return df

def main():
    """主函数"""
    print("成交量衍生特征演示")
    print("=" * 50)
    
    # 创建演示数据
    df = create_demo_data()
    
    # 演示各个特征模块
    df = demo_basic_volume_features(df)
    df = demo_volume_price_features(df)
    df = demo_mfi_feature(df)
    df = demo_turnover_rate(df)
    df = demo_comprehensive_features(df)
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n使用建议:")
    print("1. 安装TA-Lib以获得完整的MFI功能")
    print("2. 根据实际需求选择合适的特征组合")
    print("3. 结合价格特征和技术指标进行综合分析")
    print("4. 注意数据质量和特征的有效性验证")

if __name__ == "__main__":
    main() 