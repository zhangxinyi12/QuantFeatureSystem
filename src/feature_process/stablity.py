"""
特征稳定性检验模块
计算特征在不同时间窗口下的稳定性指标
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_stability(feature: pd.Series, returns: pd.Series, window: str = 'Q') -> Dict:
    """
    计算特征稳定性指标
    
    Args:
        feature: 特征序列
        returns: 收益率序列
        window: 滚动窗口 ('M'月, 'Q'季, 'Y'年)
        
    Returns:
        Dict: 包含稳定性指标的字典
    """
    try:
        # 创建数据框
        data = pd.DataFrame({
            'feature': feature,
            'next_ret': returns
        })
        
        # 按时间窗口计算IC
        ic_series = data.groupby(pd.Grouper(freq=window)).apply(
            lambda df: spearmanr(df['feature'], df['next_ret']).correlation
        ).dropna()
        
        if len(ic_series) < 2:
            logger.warning("数据不足，无法计算稳定性指标")
            return {
                'icir': np.nan,
                'max_drawdown': np.nan,
                'ic_series': ic_series,
                'is_stable': False,
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'ic_positive_ratio': np.nan
            }
        
        # 计算ICIR
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std != 0 else np.nan
        
        # 计算IC回撤
        cumulative = (1 + ic_series).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # 计算IC正比例
        ic_positive_ratio = (ic_series > 0).mean()
        
        # 稳定性判断：ICIR > 0.5 且最大回撤 > -0.3
        is_stable = (icir > 0.5) and (max_drawdown > -0.3)
        
        return {
            'icir': icir,
            'max_drawdown': max_drawdown,
            'ic_series': ic_series,
            'is_stable': is_stable,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_positive_ratio': ic_positive_ratio,
            'window': window,
            'periods': len(ic_series)
        }
        
    except Exception as e:
        logger.error(f"计算特征稳定性失败: {str(e)}")
        return {
            'icir': np.nan,
            'max_drawdown': np.nan,
            'ic_series': pd.Series(),
            'is_stable': False,
            'error': str(e)
        }

def calculate_multiple_features_stability(features_df: pd.DataFrame, returns: pd.Series, 
                                        window: str = 'Q') -> pd.DataFrame:
    """
    计算多个特征的稳定性指标
    
    Args:
        features_df: 特征数据框
        returns: 收益率序列
        window: 滚动窗口
        
    Returns:
        pd.DataFrame: 包含所有特征稳定性指标的表格
    """
    stability_results = []
    
    for feature_name in features_df.columns:
        logger.info(f"计算特征 {feature_name} 的稳定性指标...")
        
        feature = features_df[feature_name]
        stability = calculate_stability(feature, returns, window)
        
        result = {
            'feature_name': feature_name,
            'icir': stability['icir'],
            'max_drawdown': stability['max_drawdown'],
            'is_stable': stability['is_stable'],
            'ic_mean': stability.get('ic_mean', np.nan),
            'ic_std': stability.get('ic_std', np.nan),
            'ic_positive_ratio': stability.get('ic_positive_ratio', np.nan),
            'periods': stability.get('periods', 0)
        }
        
        stability_results.append(result)
    
    return pd.DataFrame(stability_results)

def filter_stable_features(features_df: pd.DataFrame, returns: pd.Series, 
                          window: str = 'Q', min_icir: float = 0.5, 
                          max_drawdown_threshold: float = -0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    筛选稳定的特征
    
    Args:
        features_df: 特征数据框
        returns: 收益率序列
        window: 滚动窗口
        min_icir: 最小ICIR阈值
        max_drawdown_threshold: 最大回撤阈值
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (稳定特征数据框, 稳定性结果数据框)
    """
    # 计算所有特征的稳定性
    stability_results = calculate_multiple_features_stability(features_df, returns, window)
    
    # 筛选稳定特征
    stable_mask = (
        (stability_results['icir'] > min_icir) & 
        (stability_results['max_drawdown'] > max_drawdown_threshold) &
        (stability_results['icir'].notna())
    )
    
    stable_features = features_df[stability_results[stable_mask]['feature_name']]
    
    logger.info(f"原始特征数量: {len(features_df.columns)}")
    logger.info(f"稳定特征数量: {len(stable_features.columns)}")
    
    if not stable_features.empty:
        logger.info(f"稳定特征: {list(stable_features.columns)}")
    
    return stable_features, stability_results

def analyze_stability_trend(feature: pd.Series, returns: pd.Series, 
                           windows: List[str] = ['M', 'Q', 'Y']) -> pd.DataFrame:
    """
    分析特征在不同时间窗口下的稳定性趋势
    
    Args:
        feature: 特征序列
        returns: 收益率序列
        windows: 时间窗口列表
        
    Returns:
        pd.DataFrame: 不同窗口下的稳定性指标
    """
    trend_results = []
    
    for window in windows:
        stability = calculate_stability(feature, returns, window)
        
        result = {
            'window': window,
            'icir': stability['icir'],
            'max_drawdown': stability['max_drawdown'],
            'is_stable': stability['is_stable'],
            'ic_mean': stability.get('ic_mean', np.nan),
            'ic_std': stability.get('ic_std', np.nan),
            'ic_positive_ratio': stability.get('ic_positive_ratio', np.nan),
            'periods': stability.get('periods', 0)
        }
        
        trend_results.append(result)
    
    return pd.DataFrame(trend_results)

def calculate_rolling_stability(feature: pd.Series, returns: pd.Series, 
                               window_size: int = 60) -> pd.Series:
    """
    计算滚动稳定性指标
    
    Args:
        feature: 特征序列
        returns: 收益率序列
        window_size: 滚动窗口大小
        
    Returns:
        pd.Series: 滚动ICIR序列
    """
    rolling_icir = pd.Series(index=feature.index, dtype=float)
    
    for i in range(window_size, len(feature)):
        window_feature = feature.iloc[i-window_size:i]
        window_returns = returns.iloc[i-window_size:i]
        
        if len(window_feature.dropna()) >= window_size * 0.8:  # 至少80%的数据
            try:
                ic = spearmanr(window_feature, window_returns).correlation
                if not np.isnan(ic):
                    # 计算滚动ICIR（简化版本）
                    rolling_icir.iloc[i] = ic
            except:
                continue
    
    return rolling_icir

def generate_stability_report(features_df: pd.DataFrame, returns: pd.Series, 
                             window: str = 'Q') -> str:
    """
    生成特征稳定性报告
    
    Args:
        features_df: 特征数据框
        returns: 收益率序列
        window: 滚动窗口
        
    Returns:
        str: 稳定性报告
    """
    stability_results = calculate_multiple_features_stability(features_df, returns, window)
    
    report = f"""
特征稳定性分析报告
==================

分析窗口: {window}
特征总数: {len(features_df.columns)}
分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

稳定性统计:
- 稳定特征数量: {stability_results['is_stable'].sum()}
- 稳定特征比例: {stability_results['is_stable'].mean():.2%}

ICIR统计:
- 平均ICIR: {stability_results['icir'].mean():.4f}
- 中位数ICIR: {stability_results['icir'].median():.4f}
- 最大ICIR: {stability_results['icir'].max():.4f}
- 最小ICIR: {stability_results['icir'].min():.4f}

最大回撤统计:
- 平均最大回撤: {stability_results['max_drawdown'].mean():.4f}
- 中位数最大回撤: {stability_results['max_drawdown'].median():.4f}

IC正比例统计:
- 平均IC正比例: {stability_results['ic_positive_ratio'].mean():.2%}

稳定特征列表:
"""
    
    stable_features = stability_results[stability_results['is_stable']]
    if not stable_features.empty:
        for _, row in stable_features.iterrows():
            report += f"- {row['feature_name']}: ICIR={row['icir']:.4f}, 最大回撤={row['max_drawdown']:.4f}\n"
    else:
        report += "- 无稳定特征\n"
    
    return report

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟特征和收益率数据
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    feature = pd.Series(np.random.randn(n_samples), index=dates)
    returns = pd.Series(np.random.randn(n_samples) * 0.02, index=dates)
    
    # 计算单个特征稳定性
    stability = calculate_stability(feature, returns, 'Q')
    print("单个特征稳定性结果:")
    print(f"ICIR: {stability['icir']:.4f}")
    print(f"最大回撤: {stability['max_drawdown']:.4f}")
    print(f"是否稳定: {stability['is_stable']}")
    
    # 多个特征示例
    features_df = pd.DataFrame({
        'momentum': np.random.randn(n_samples),
        'volatility': np.random.randn(n_samples),
        'volume': np.random.randn(n_samples)
    }, index=dates)
    
    # 计算多个特征稳定性
    multi_results = calculate_multiple_features_stability(features_df, returns, 'Q')
    print("\n多个特征稳定性结果:")
    print(multi_results[['feature_name', 'icir', 'max_drawdown', 'is_stable']])
    
    # 筛选稳定特征
    stable_features, all_results = filter_stable_features(features_df, returns, 'Q')
    print(f"\n稳定特征数量: {len(stable_features.columns)}")
    
    # 生成报告
    report = generate_stability_report(features_df, returns, 'Q')
    print("\n稳定性报告:")
    print(report)