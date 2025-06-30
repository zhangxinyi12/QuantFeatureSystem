#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测运行脚本
使用预定义的配置运行回测
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.backtest_config import (
    get_backtest_period, 
    get_backtest_params, 
    list_backtest_periods, 
    list_backtest_params,
    BACKTEST_PERIODS,
    BACKTEST_PARAMS
)
from src.main import QuantFeatureSystem

def run_backtest(period_name, params_name='base', use_ssh_tunnel=False):
    """运行回测
    
    Args:
        period_name: 回测周期名称
        params_name: 参数配置名称
        use_ssh_tunnel: 是否使用SSH隧道
    """
    print("="*60)
    print("开始运行回测")
    print("="*60)
    
    # 获取回测配置
    try:
        period = get_backtest_period(period_name)
        params = get_backtest_params(params_name)
    except ValueError as e:
        print(f"配置错误: {e}")
        return False
    
    print(f"回测周期: {period['name']}")
    print(f"时间范围: {period['start_date']} 到 {period['end_date']}")
    print(f"参数配置: {params_name}")
    print(f"描述: {period['description']}")
    print()
    
    # 创建系统实例
    system = QuantFeatureSystem(
        start_date=period['start_date'],
        end_date=period['end_date']
    )
    
    # 创建模拟的args对象
    class MockArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = MockArgs(**params)
    
    # 运行回测
    try:
        result = system.process_stock_data(args, use_ssh_tunnel=use_ssh_tunnel)
        
        if result is not None:
            print(f"\n✅ 回测完成！")
            print(f"处理数据行数: {len(result):,}")
            return True
        else:
            print(f"\n❌ 回测失败！")
            return False
            
    except Exception as e:
        print(f"\n❌ 回测出错: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='回测运行脚本')
    
    # 基本参数
    parser.add_argument('--period', type=str, default='medium_term', 
                       help='回测周期名称')
    parser.add_argument('--params', type=str, default='base',
                       help='参数配置名称')
    parser.add_argument('--use-ssh-tunnel', action='store_true',
                       help='使用SSH隧道连接数据库')
    
    # 信息显示参数
    parser.add_argument('--list-periods', action='store_true',
                       help='列出所有可用的回测周期')
    parser.add_argument('--list-params', action='store_true',
                       help='列出所有可用的参数配置')
    
    args = parser.parse_args()
    
    # 显示配置信息
    if args.list_periods:
        list_backtest_periods()
        return 0
    
    if args.list_params:
        list_backtest_params()
        return 0
    
    # 验证参数
    if args.period not in BACKTEST_PERIODS:
        print(f"❌ 未知的回测周期: {args.period}")
        print("使用 --list-periods 查看可用的回测周期")
        return 1
    
    if args.params not in BACKTEST_PARAMS:
        print(f"❌ 未知的参数配置: {args.params}")
        print("使用 --list-params 查看可用的参数配置")
        return 1
    
    # 运行回测
    success = run_backtest(
        period_name=args.period,
        params_name=args.params,
        use_ssh_tunnel=args.use_ssh_tunnel
    )
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main()) 