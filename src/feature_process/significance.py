import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_significance(feature, returns, confidence=0.99, min_ic_threshold=0.05):
    """
    计算特征统计显著性
    
    Args:
        feature: 特征值序列
        returns: 未来收益率序列
        confidence: 置信水平 (0.99对应p<0.01)
        min_ic_threshold: 最小IC阈值 (默认0.05)
        
    Returns:
        dict: 包含IC值、p值、置信区间、显著性判断等
    """
    # 数据预处理
    feature = np.array(feature)
    returns = np.array(returns)
    
    # 移除缺失值
    valid_mask = ~(np.isnan(feature) | np.isnan(returns))
    feature_clean = feature[valid_mask]
    returns_clean = returns[valid_mask]
    
    if len(feature_clean) < 30:
        return {
            'ic': np.nan,
            'p_value': 1.0,
            'confidence_interval': (np.nan, np.nan),
            'is_significant': False,
            'sample_size': len(feature_clean),
            'error': '样本量不足'
        }
    
    # 计算Rank IC (Spearman相关系数)
    try:
        ic_result = spearmanr(feature_clean, returns_clean)
        ic = ic_result.correlation
        p_value = ic_result.pvalue
    except:
        ic = np.nan
        p_value = 1.0
    
    # Bootstrap显著性检验
    n_bootstrap = 10000  # 增加bootstrap次数提高精度
    bootstrap_ics = []
    
    for _ in range(n_bootstrap):
        try:
            # 随机抽样（有放回）
            idx = np.random.choice(len(feature_clean), len(feature_clean), replace=True)
            bootstrap_ic = spearmanr(feature_clean[idx], returns_clean[idx]).correlation
            if not np.isnan(bootstrap_ic):
                bootstrap_ics.append(bootstrap_ic)
        except:
            continue
    
    if len(bootstrap_ics) < 1000:
        return {
            'ic': ic,
            'p_value': p_value,
            'confidence_interval': (np.nan, np.nan),
            'is_significant': False,
            'sample_size': len(feature_clean),
            'error': 'Bootstrap失败'
        }
    
    # 计算置信区间
    alpha = 1 - confidence
    lower_bound = np.percentile(bootstrap_ics, alpha * 50)
    upper_bound = np.percentile(bootstrap_ics, 100 - alpha * 50)
    
    # 计算bootstrap p值
    bootstrap_p_value = np.mean(np.array(bootstrap_ics) <= 0) if ic > 0 else np.mean(np.array(bootstrap_ics) >= 0)
    bootstrap_p_value = min(bootstrap_p_value, 1 - bootstrap_p_value) * 2  # 双尾检验
    
    # 判断显著性：IC > 0.05 且 p值 < 0.01
    is_significant = (abs(ic) > min_ic_threshold) and (bootstrap_p_value < 0.01)
    
    return {
        'ic': ic,
        'p_value': bootstrap_p_value,
        'confidence_interval': (lower_bound, upper_bound),
        'is_significant': is_significant,
        'sample_size': len(feature_clean),
        'bootstrap_count': len(bootstrap_ics),
        'ic_threshold': min_ic_threshold,
        'p_threshold': 0.01
    }

def calculate_multiple_features_significance(features_df, returns_series, confidence=0.99, min_ic_threshold=0.05):
    """
    计算多个特征的显著性
    
    Args:
        features_df: 特征DataFrame，每列为一个特征
        returns_series: 未来收益率Series
        confidence: 置信水平
        min_ic_threshold: 最小IC阈值
        
    Returns:
        DataFrame: 每个特征的显著性检验结果
    """
    results = []
    
    for feature_name in features_df.columns:
        feature_values = features_df[feature_name].values
        result = calculate_significance(feature_values, returns_series.values, confidence, min_ic_threshold)
        result['feature_name'] = feature_name
        results.append(result)
    
    return pd.DataFrame(results)

def filter_significant_features(features_df, returns_series, confidence=0.99, min_ic_threshold=0.05):
    """
    筛选显著特征
    
    Args:
        features_df: 特征DataFrame
        returns_series: 未来收益率Series
        confidence: 置信水平
        min_ic_threshold: 最小IC阈值
        
    Returns:
        tuple: (显著特征DataFrame, 显著性结果DataFrame)
    """
    # 计算所有特征的显著性
    significance_results = calculate_multiple_features_significance(
        features_df, returns_series, confidence, min_ic_threshold
    )
    
    # 筛选显著特征
    significant_features = significance_results[
        (significance_results['is_significant'] == True) & 
        (significance_results['ic'].notna())
    ]
    
    # 获取显著特征名称
    significant_feature_names = significant_features['feature_name'].tolist()
    
    # 返回显著特征和结果
    if significant_feature_names:
        significant_features_df = features_df[significant_feature_names]
    else:
        significant_features_df = pd.DataFrame()
    
    return significant_features_df, significance_results

def calculate_ic_decay(feature, returns_forward, max_lag=20):
    """
    计算IC衰减
    
    Args:
        feature: 特征值
        returns_forward: 未来多期收益率DataFrame，列为lag期数
        max_lag: 最大滞后期数
        
    Returns:
        DataFrame: IC衰减结果
    """
    ic_decay_results = []
    
    for lag in range(1, max_lag + 1):
        if f'return_{lag}d' in returns_forward.columns:
            future_returns = returns_forward[f'return_{lag}d']
            result = calculate_significance(feature, future_returns)
            result['lag'] = lag
            ic_decay_results.append(result)
    
    return pd.DataFrame(ic_decay_results)

# 使用示例
if __name__ == "__main__":
    # 示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成模拟特征和收益率
    feature = np.random.randn(n_samples)
    returns = 0.1 * feature + 0.05 * np.random.randn(n_samples)  # 正相关
    
    # 计算单个特征显著性
    significance = calculate_significance(feature, returns)
    print("单个特征显著性检验结果:")
    print(f"IC: {significance['ic']:.4f}")
    print(f"P值: {significance['p_value']:.4f}")
    print(f"是否显著: {significance['is_significant']}")
    print(f"置信区间: ({significance['confidence_interval'][0]:.4f}, {significance['confidence_interval'][1]:.4f})")
    
    # 多个特征示例
    features_df = pd.DataFrame({
        'momentum': np.random.randn(n_samples),
        'volatility': np.random.randn(n_samples),
        'volume': np.random.randn(n_samples)
    })
    
    # 计算多个特征显著性
    multi_results = calculate_multiple_features_significance(features_df, pd.Series(returns))
    print("\n多个特征显著性检验结果:")
    print(multi_results[['feature_name', 'ic', 'p_value', 'is_significant']])
    
    # 筛选显著特征
    significant_features, all_results = filter_significant_features(features_df, pd.Series(returns))
    print(f"\n显著特征数量: {len(significant_features.columns)}")
    if not significant_features.empty:
        print(f"显著特征: {list(significant_features.columns)}")