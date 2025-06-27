#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据存储功能使用示例
展示如何使用存储管理器、分区管理和存储策略
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_sample_data():
    """创建示例数据"""
    print("创建示例数据...")
    
    # 生成模拟的股票数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    market_codes = [1, 2]  # 沪深市场
    
    data = []
    for date in dates:
        for market_code in market_codes:
            # 生成价格数据
            base_price = 100 + np.random.normal(0, 10)
            open_price = base_price
            close_price = base_price + np.random.normal(0, 2)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 1))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 1))
            volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'date': date,
                'market_code': market_code,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
    
    df = pd.DataFrame(data)
    print(f"数据形状: {df.shape}")
    print(f"数据范围: {df['date'].min()} 到 {df['date'].max()}")
    return df

def demo_storage_config():
    """演示存储配置功能"""
    print("\n=== 演示存储配置功能 ===")
    
    try:
        from src.storage import StorageConfig, StorageStrategy
        
        # 创建存储配置
        config = StorageConfig()
        
        # 查看默认策略
        default_strategy = config.get_strategy('default')
        print(f"✓ 默认策略: {default_strategy.format}, 压缩: {default_strategy.compression}")
        
        # 查看高性能策略
        perf_strategy = config.get_strategy('high_performance')
        print(f"✓ 高性能策略: {perf_strategy.format}, 压缩: {perf_strategy.compression}")
        
        # 查看支持的格式信息
        formats_info = config.get_formats_info()
        print("✓ 支持的存储格式:")
        for fmt, info in formats_info.items():
            print(f"  - {fmt}: {info['description']}")
        
        # 推荐存储策略
        recommendation = config.recommend_strategy(500000, 'column')
        print(f"✓ 推荐策略 (50万行, 列查询): {recommendation}")
        
        return config
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        return None

def demo_data_manager():
    """演示数据管理器功能"""
    print("\n=== 演示数据管理器功能 ===")
    
    try:
        from src.storage import DataManager
        
        # 创建数据管理器
        data_manager = DataManager(base_path="output/demo_storage")
        
        # 创建示例数据
        df = create_sample_data()
        
        # 保存原始数据
        print("保存原始数据...")
        raw_path = data_manager.save_data(
            df, 
            data_type='raw',
            strategy_name='raw_data',
            partition_by=['date', 'market_code'],
            metadata={'source': 'demo', 'description': '示例原始数据'}
        )
        print(f"✓ 原始数据保存到: {raw_path}")
        
        # 保存特征数据（模拟）
        feature_df = df.copy()
        feature_df['price_change'] = feature_df['close'].pct_change()
        feature_df['volume_ma'] = feature_df['volume'].rolling(5).mean()
        
        print("保存特征数据...")
        feature_path = data_manager.save_data(
            feature_df,
            data_type='feature',
            strategy_name='feature_data',
            partition_by=['date', 'feature_type'],
            metadata={'source': 'demo', 'description': '示例特征数据'}
        )
        print(f"✓ 特征数据保存到: {feature_path}")
        
        # 保存处理后的数据
        print("保存处理后数据...")
        processed_path = data_manager.save_data(
            feature_df.dropna(),
            data_type='processed',
            strategy_name='high_performance',
            metadata={'source': 'demo', 'description': '示例处理后数据'}
        )
        print(f"✓ 处理后数据保存到: {processed_path}")
        
        return data_manager, raw_path, feature_path, processed_path
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        return None, None, None, None

def demo_data_loading(data_manager, raw_path, feature_path, processed_path):
    """演示数据加载功能"""
    print("\n=== 演示数据加载功能 ===")
    
    if data_manager is None:
        print("✗ 数据管理器未初始化")
        return
    
    try:
        # 加载原始数据
        print("加载原始数据...")
        raw_data = data_manager.load_data(raw_path)
        print(f"✓ 原始数据加载完成: {raw_data.shape}")
        
        # 加载特征数据
        print("加载特征数据...")
        feature_data = data_manager.load_data(feature_path)
        print(f"✓ 特征数据加载完成: {feature_data.shape}")
        
        # 加载处理后数据
        print("加载处理后数据...")
        processed_data = data_manager.load_data(processed_path)
        print(f"✓ 处理后数据加载完成: {processed_data.shape}")
        
        # 验证数据一致性
        print("验证数据一致性...")
        assert len(raw_data) == len(feature_data), "数据行数不一致"
        print("✓ 数据一致性验证通过")
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")

def demo_storage_info(data_manager):
    """演示存储信息功能"""
    print("\n=== 演示存储信息功能 ===")
    
    if data_manager is None:
        print("✗ 数据管理器未初始化")
        return
    
    try:
        # 获取存储信息
        storage_info = data_manager.get_storage_info()
        
        print("✓ 存储信息:")
        print(f"  - 基础路径: {storage_info['base_path']}")
        print(f"  - 文件总数: {storage_info['file_count']}")
        print(f"  - 总大小: {storage_info['total_size'] / 1024 / 1024:.2f} MB")
        print(f"  - 数据类型: {storage_info['data_types']}")
        print(f"  - 可用策略: {storage_info['strategies']}")
        
        # 清理临时文件
        print("清理临时文件...")
        data_manager.cleanup_temporary(max_age_days=1)
        print("✓ 临时文件清理完成")
        
    except Exception as e:
        print(f"✗ 获取存储信息失败: {e}")

def demo_custom_strategy():
    """演示自定义存储策略"""
    print("\n=== 演示自定义存储策略 ===")
    
    try:
        from src.storage import StorageConfig, StorageStrategy
        
        config = StorageConfig()
        
        # 创建自定义策略
        custom_strategy = StorageStrategy(
            format='parquet',
            compression='lz4',
            partition_by=['date'],
            partition_size=100000,
            chunk_size=5000,
            include_metadata=True
        )
        
        # 添加自定义策略
        config.add_strategy('custom_daily', custom_strategy)
        
        # 使用自定义策略
        strategy = config.get_strategy('custom_daily')
        print(f"✓ 自定义策略: {strategy.format}, 分区: {strategy.partition_by}")
        
        return config
        
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        return None

def main():
    """主函数"""
    print("数据存储功能演示")
    print("=" * 50)
    
    # 演示存储配置
    config = demo_storage_config()
    
    # 演示数据管理器
    data_manager, raw_path, feature_path, processed_path = demo_data_manager()
    
    # 演示数据加载
    demo_data_loading(data_manager, raw_path, feature_path, processed_path)
    
    # 演示存储信息
    demo_storage_info(data_manager)
    
    # 演示自定义策略
    demo_custom_strategy()
    
    print("\n" + "=" * 50)
    print("数据存储功能演示完成")
    print("生成的文件位于: output/demo_storage/")

if __name__ == "__main__":
    main() 