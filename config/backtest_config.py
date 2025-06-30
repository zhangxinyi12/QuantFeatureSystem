#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测配置文件
定义不同的回测时间范围和参数
"""

# 回测时间范围配置
BACKTEST_PERIODS = {
    # 短期回测（用于快速验证）
    'short_term': {
        'name': '短期回测',
        'start_date': '2024-01-01',
        'end_date': '2024-03-31',
        'description': '2024年第一季度，用于快速验证策略'
    },
    
    # 中期回测（标准回测）
    'medium_term': {
        'name': '中期回测',
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'description': '2024年全年，标准回测周期'
    },
    
    # 长期回测（历史验证）
    'long_term': {
        'name': '长期回测',
        'start_date': '2023-01-01',
        'end_date': '2024-12-31',
        'description': '2023-2024年，长期历史验证'
    },
    
    # 牛市回测
    'bull_market': {
        'name': '牛市回测',
        'start_date': '2020-03-01',
        'end_date': '2021-12-31',
        'description': '2020-2021年牛市期间'
    },
    
    # 熊市回测
    'bear_market': {
        'name': '熊市回测',
        'start_date': '2022-01-01',
        'end_date': '2022-12-31',
        'description': '2022年熊市期间'
    },
    
    # 震荡市回测
    'sideways_market': {
        'name': '震荡市回测',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'description': '2023年震荡市期间'
    },
    
    # 最近数据回测
    'recent': {
        'name': '最近数据回测',
        'start_date': '2024-10-01',
        'end_date': '2024-12-31',
        'description': '2024年第四季度，最近数据'
    }
}

# 默认回测配置
DEFAULT_BACKTEST = 'medium_term'

# 回测参数配置
BACKTEST_PARAMS = {
    # 基础参数
    'base': {
        'market_codes': [83, 90],  # 深交所、上交所
        'adj_type': 'forward',     # 前复权
        'include_technical': True, # 包含技术指标
        'include_features': True,  # 包含量化特征
        'feature_types': ['price', 'volume', 'technical'],
        'output_format': 'parquet'
    },
    
    # 快速测试参数
    'quick_test': {
        'market_codes': [83, 90],
        'adj_type': 'none',
        'include_technical': False,
        'include_features': False,
        'feature_types': [],
        'output_format': 'csv'
    },
    
    # 完整特征参数
    'full_features': {
        'market_codes': [83, 90],
        'adj_type': 'forward',
        'include_technical': True,
        'include_features': True,
        'feature_types': ['price', 'volume', 'technical', 'cross_asset'],
        'output_format': 'parquet'
    }
}

def get_backtest_period(period_name=None):
    """获取回测时间范围
    
    Args:
        period_name: 回测周期名称，如果为None则使用默认配置
        
    Returns:
        dict: 包含start_date和end_date的字典
    """
    if period_name is None:
        period_name = DEFAULT_BACKTEST
    
    if period_name not in BACKTEST_PERIODS:
        raise ValueError(f"未知的回测周期: {period_name}")
    
    period = BACKTEST_PERIODS[period_name]
    return {
        'start_date': period['start_date'],
        'end_date': period['end_date'],
        'name': period['name'],
        'description': period['description']
    }

def get_backtest_params(params_name='base'):
    """获取回测参数
    
    Args:
        params_name: 参数配置名称
        
    Returns:
        dict: 回测参数字典
    """
    if params_name not in BACKTEST_PARAMS:
        raise ValueError(f"未知的参数配置: {params_name}")
    
    return BACKTEST_PARAMS[params_name]

def list_backtest_periods():
    """列出所有可用的回测周期"""
    print("可用的回测周期:")
    print("=" * 60)
    for name, config in BACKTEST_PERIODS.items():
        print(f"{name:15} | {config['name']:10} | {config['start_date']} 到 {config['end_date']}")
        print(f"{'':15} | {'':10} | {config['description']}")
        print()

def list_backtest_params():
    """列出所有可用的参数配置"""
    print("可用的参数配置:")
    print("=" * 60)
    for name, params in BACKTEST_PARAMS.items():
        print(f"{name}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print()

if __name__ == '__main__':
    # 测试配置
    print("回测配置测试")
    print("=" * 60)
    
    # 列出所有配置
    list_backtest_periods()
    list_backtest_params()
    
    # 测试获取配置
    period = get_backtest_period('short_term')
    print(f"短期回测配置: {period}")
    
    params = get_backtest_params('base')
    print(f"基础参数配置: {params}") 