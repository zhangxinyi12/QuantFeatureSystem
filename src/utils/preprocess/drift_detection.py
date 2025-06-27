#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征漂移检测工具模块
提供数据漂移检测、概念漂移检测、分布变化监控等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

class DriftDetector:
    """特征漂移检测器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化漂移检测器
        
        参数:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.drift_history = []
        self.reference_data = None
        
    def set_reference_data(self, reference_df: pd.DataFrame, 
                          feature_columns: List[str] = None):
        """
        设置参考数据
        
        参数:
            reference_df: 参考数据DataFrame
            feature_columns: 特征列列表
        """
        if feature_columns is None:
            feature_columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()
            
        self.reference_data = reference_df[feature_columns].copy()
        self.logger.info(f"已设置参考数据，包含 {len(feature_columns)} 个特征")
        
    def detect_distribution_drift(self, current_df: pd.DataFrame,
                                feature_columns: List[str] = None,
                                method: str = 'ks_test',
                                alpha: float = 0.05) -> Dict:
        """
        检测分布漂移
        
        参数:
            current_df: 当前数据DataFrame
            feature_columns: 特征列列表
            method: 检测方法 ('ks_test', 'chi2_test', 'wasserstein')
            alpha: 显著性水平
            
        返回:
            漂移检测结果
        """
        if self.reference_data is None:
            raise ValueError("请先设置参考数据")
            
        if feature_columns is None:
            feature_columns = self.reference_data.columns.tolist()
            
        results = {
            'drift_detected': False,
            'drift_features': [],
            'drift_scores': {},
            'p_values': {},
            'method': method,
            'alpha': alpha,
            'timestamp': datetime.now()
        }
        
        for feature in feature_columns:
            if feature not in self.reference_data.columns or feature not in current_df.columns:
                continue
                
            ref_data = self.reference_data[feature].dropna()
            cur_data = current_df[feature].dropna()
            
            if len(ref_data) == 0 or len(cur_data) == 0:
                continue
                
            if method == 'ks_test':
                drift_result = self._ks_test_drift(ref_data, cur_data, alpha)
            elif method == 'chi2_test':
                drift_result = self._chi2_test_drift(ref_data, cur_data, alpha)
            elif method == 'wasserstein':
                drift_result = self._wasserstein_drift(ref_data, cur_data)
            else:
                raise ValueError(f"不支持的检测方法: {method}")
                
            results['drift_scores'][feature] = drift_result['score']
            results['p_values'][feature] = drift_result.get('p_value', None)
            
            if drift_result['drift_detected']:
                results['drift_detected'] = True
                results['drift_features'].append(feature)
                
        # 记录检测历史
        self.drift_history.append(results)
        
        return results
        
    def _ks_test_drift(self, ref_data: pd.Series, cur_data: pd.Series, alpha: float) -> Dict:
        """KS检验检测漂移"""
        statistic, p_value = ks_2samp(ref_data, cur_data)
        
        drift_detected = p_value < alpha
        score = 1 - p_value  # 转换为漂移分数
        
        return {
            'drift_detected': drift_detected,
            'score': score,
            'p_value': p_value,
            'statistic': statistic
        }
        
    def _chi2_test_drift(self, ref_data: pd.Series, cur_data: pd.Series, alpha: float) -> Dict:
        """卡方检验检测漂移"""
        # 创建分箱
        bins = np.linspace(min(ref_data.min(), cur_data.min()),
                          max(ref_data.max(), cur_data.max()), 10)
        
        ref_hist, _ = np.histogram(ref_data, bins=bins)
        cur_hist, _ = np.histogram(cur_data, bins=bins)
        
        # 确保没有零频次
        ref_hist = ref_hist + 1e-10
        cur_hist = cur_hist + 1e-10
        
        # 卡方检验
        chi2_stat, p_value, _, _ = chi2_contingency([ref_hist, cur_hist])
        
        drift_detected = p_value < alpha
        score = 1 - p_value
        
        return {
            'drift_detected': drift_detected,
            'score': score,
            'p_value': p_value,
            'statistic': chi2_stat
        }
        
    def _wasserstein_drift(self, ref_data: pd.Series, cur_data: pd.Series) -> Dict:
        """Wasserstein距离检测漂移"""
        from scipy.stats import wasserstein_distance
        
        distance = wasserstein_distance(ref_data, cur_data)
        
        # 标准化距离
        ref_std = ref_data.std()
        normalized_distance = distance / ref_std if ref_std > 0 else 0
        
        # 设置阈值（可以根据实际情况调整）
        threshold = 0.1
        drift_detected = normalized_distance > threshold
        score = min(normalized_distance / threshold, 1.0)
        
        return {
            'drift_detected': drift_detected,
            'score': score,
            'distance': distance,
            'normalized_distance': normalized_distance
        }
        
    def detect_covariate_drift(self, current_df: pd.DataFrame,
                              feature_columns: List[str] = None,
                              method: str = 'pca') -> Dict:
        """
        检测协变量漂移
        
        参数:
            current_df: 当前数据DataFrame
            feature_columns: 特征列列表
            method: 检测方法 ('pca', 'density_ratio')
            
        返回:
            协变量漂移检测结果
        """
        if self.reference_data is None:
            raise ValueError("请先设置参考数据")
            
        if feature_columns is None:
            feature_columns = self.reference_data.columns.tolist()
            
        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'method': method,
            'timestamp': datetime.now()
        }
        
        if method == 'pca':
            drift_result = self._pca_covariate_drift(current_df, feature_columns)
        elif method == 'density_ratio':
            drift_result = self._density_ratio_covariate_drift(current_df, feature_columns)
        else:
            raise ValueError(f"不支持的检测方法: {method}")
            
        results.update(drift_result)
        self.drift_history.append(results)
        
        return results
        
    def _pca_covariate_drift(self, current_df: pd.DataFrame, feature_columns: List[str]) -> Dict:
        """基于PCA的协变量漂移检测"""
        # 准备数据
        ref_data = self.reference_data[feature_columns].dropna()
        cur_data = current_df[feature_columns].dropna()
        
        if len(ref_data) == 0 or len(cur_data) == 0:
            return {'drift_detected': False, 'drift_score': 0.0}
            
        # 标准化
        scaler = StandardScaler()
        ref_scaled = scaler.fit_transform(ref_data)
        cur_scaled = scaler.transform(cur_data)
        
        # PCA降维
        pca = PCA(n_components=min(3, len(feature_columns)))
        ref_pca = pca.fit_transform(ref_scaled)
        cur_pca = pca.transform(cur_scaled)
        
        # 计算分布差异
        drift_scores = []
        for i in range(ref_pca.shape[1]):
            ref_comp = ref_pca[:, i]
            cur_comp = cur_pca[:, i]
            
            statistic, p_value = ks_2samp(ref_comp, cur_comp)
            drift_scores.append(1 - p_value)
            
        # 综合漂移分数
        overall_score = np.mean(drift_scores)
        drift_detected = overall_score > 0.8  # 阈值可调整
        
        return {
            'drift_detected': drift_detected,
            'drift_score': overall_score,
            'component_scores': drift_scores
        }
        
    def _density_ratio_covariate_drift(self, current_df: pd.DataFrame, feature_columns: List[str]) -> Dict:
        """基于密度比的协变量漂移检测"""
        # 简化的密度比估计
        ref_data = self.reference_data[feature_columns].dropna()
        cur_data = current_df[feature_columns].dropna()
        
        if len(ref_data) == 0 or len(cur_data) == 0:
            return {'drift_detected': False, 'drift_score': 0.0}
            
        # 计算每个特征的密度比
        density_ratios = []
        for feature in feature_columns:
            ref_feature = ref_data[feature]
            cur_feature = cur_data[feature]
            
            # 使用核密度估计
            ref_kde = stats.gaussian_kde(ref_feature)
            cur_kde = stats.gaussian_kde(cur_feature)
            
            # 在参考数据上计算密度比
            sample_points = np.linspace(ref_feature.min(), ref_feature.max(), 100)
            ref_density = ref_kde(sample_points)
            cur_density = cur_kde(sample_points)
            
            # 避免除零
            density_ratio = np.mean(cur_density / (ref_density + 1e-10))
            density_ratios.append(density_ratio)
            
        # 计算漂移分数
        drift_score = np.std(density_ratios)  # 密度比的标准差作为漂移指标
        drift_detected = drift_score > 0.5  # 阈值可调整
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'density_ratios': density_ratios
        }
        
    def detect_concept_drift(self, reference_labels: pd.Series,
                           current_labels: pd.Series,
                           reference_features: pd.DataFrame,
                           current_features: pd.DataFrame,
                           method: str = 'performance_drop') -> Dict:
        """
        检测概念漂移
        
        参数:
            reference_labels: 参考标签
            current_labels: 当前标签
            reference_features: 参考特征
            current_features: 当前特征
            method: 检测方法 ('performance_drop', 'distribution_shift')
            
        返回:
            概念漂移检测结果
        """
        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'method': method,
            'timestamp': datetime.now()
        }
        
        if method == 'performance_drop':
            drift_result = self._performance_drop_concept_drift(
                reference_labels, current_labels, reference_features, current_features
            )
        elif method == 'distribution_shift':
            drift_result = self._distribution_shift_concept_drift(
                reference_labels, current_labels, reference_features, current_features
            )
        else:
            raise ValueError(f"不支持的检测方法: {method}")
            
        results.update(drift_result)
        self.drift_history.append(results)
        
        return results
        
    def _performance_drop_concept_drift(self, reference_labels: pd.Series,
                                      current_labels: pd.Series,
                                      reference_features: pd.DataFrame,
                                      current_features: pd.DataFrame) -> Dict:
        """基于性能下降的概念漂移检测"""
        # 这里简化处理，实际应用中需要训练模型
        # 计算标签分布的变化
        ref_label_dist = reference_labels.value_counts(normalize=True)
        cur_label_dist = current_labels.value_counts(normalize=True)
        
        # 计算分布差异
        common_labels = set(ref_label_dist.index) & set(cur_label_dist.index)
        if not common_labels:
            return {'drift_detected': True, 'drift_score': 1.0}
            
        drift_score = 0.0
        for label in common_labels:
            ref_prob = ref_label_dist.get(label, 0)
            cur_prob = cur_label_dist.get(label, 0)
            drift_score += abs(ref_prob - cur_prob)
            
        drift_score /= len(common_labels)
        drift_detected = drift_score > 0.2  # 阈值可调整
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score
        }
        
    def _distribution_shift_concept_drift(self, reference_labels: pd.Series,
                                        current_labels: pd.Series,
                                        reference_features: pd.DataFrame,
                                        current_features: pd.DataFrame) -> Dict:
        """基于分布偏移的概念漂移检测"""
        # 计算特征空间中的分布差异
        ref_data = reference_features.select_dtypes(include=[np.number])
        cur_data = current_features.select_dtypes(include=[np.number])
        
        if ref_data.empty or cur_data.empty:
            return {'drift_detected': False, 'drift_score': 0.0}
            
        # 使用PCA降维后计算分布差异
        scaler = StandardScaler()
        ref_scaled = scaler.fit_transform(ref_data)
        cur_scaled = scaler.transform(cur_data)
        
        pca = PCA(n_components=min(3, ref_data.shape[1]))
        ref_pca = pca.fit_transform(ref_scaled)
        cur_pca = pca.transform(cur_scaled)
        
        # 计算每个主成分的分布差异
        drift_scores = []
        for i in range(ref_pca.shape[1]):
            statistic, p_value = ks_2samp(ref_pca[:, i], cur_pca[:, i])
            drift_scores.append(1 - p_value)
            
        overall_score = np.mean(drift_scores)
        drift_detected = overall_score > 0.8
        
        return {
            'drift_detected': drift_detected,
            'drift_score': overall_score,
            'component_scores': drift_scores
        }
        
    def get_drift_summary(self, hours: int = 24) -> Dict:
        """
        获取漂移检测摘要
        
        参数:
            hours: 统计最近多少小时的数据
            
        返回:
            漂移检测摘要
        """
        if not self.drift_history:
            return {}
            
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_drifts = [d for d in self.drift_history if d['timestamp'] >= cutoff_time]
        
        if not recent_drifts:
            return {}
            
        summary = {
            'total_detections': len(recent_drifts),
            'drift_detected_count': sum(1 for d in recent_drifts if d.get('drift_detected', False)),
            'drift_rate': sum(1 for d in recent_drifts if d.get('drift_detected', False)) / len(recent_drifts),
            'avg_drift_score': np.mean([d.get('drift_score', 0) for d in recent_drifts]),
            'max_drift_score': max([d.get('drift_score', 0) for d in recent_drifts]),
            'most_drifted_features': self._get_most_drifted_features(recent_drifts)
        }
        
        return summary
        
    def _get_most_drifted_features(self, drifts: List[Dict]) -> List[str]:
        """获取漂移最频繁的特征"""
        feature_drift_counts = {}
        
        for drift in drifts:
            if 'drift_features' in drift:
                for feature in drift['drift_features']:
                    feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1
                    
        # 按漂移次数排序
        sorted_features = sorted(feature_drift_counts.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return [feature for feature, count in sorted_features[:5]]  # 返回前5个
        
    def plot_drift_history(self, save_path: Optional[str] = None) -> str:
        """
        绘制漂移检测历史
        
        参数:
            save_path: 保存路径
            
        返回:
            图表保存路径
        """
        if not self.drift_history:
            self.logger.warning("没有漂移检测历史数据")
            return ""
            
        # 准备数据
        timestamps = [d['timestamp'] for d in self.drift_history]
        drift_scores = [d.get('drift_score', 0) for d in self.drift_history]
        drift_detected = [d.get('drift_detected', False) for d in self.drift_history]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 漂移分数时间序列
        ax1.plot(timestamps, drift_scores, 'b-', alpha=0.7)
        ax1.set_title('漂移分数时间序列')
        ax1.set_ylabel('漂移分数')
        ax1.grid(True, alpha=0.3)
        
        # 漂移检测结果
        drift_dates = [timestamps[i] for i, detected in enumerate(drift_detected) if detected]
        ax2.scatter(drift_dates, [1] * len(drift_dates), c='red', alpha=0.7, s=50)
        ax2.set_title('漂移检测结果')
        ax2.set_ylabel('漂移检测')
        ax2.set_ylim(0, 2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"drift_history_{timestamp}.png"
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"漂移历史图表已保存到: {save_path}")
        return save_path

# 便捷函数
def quick_drift_detection(reference_df: pd.DataFrame,
                         current_df: pd.DataFrame,
                         feature_columns: List[str] = None,
                         method: str = 'ks_test') -> Dict:
    """
    快速漂移检测
    
    参数:
        reference_df: 参考数据
        current_df: 当前数据
        feature_columns: 特征列
        method: 检测方法
        
    返回:
        漂移检测结果
    """
    detector = DriftDetector()
    detector.set_reference_data(reference_df, feature_columns)
    return detector.detect_distribution_drift(current_df, feature_columns, method)

# 示例使用
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # 参考数据
    ref_data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(10, 2, 100),
        'feature3': np.random.exponential(1, 100)
    }
    reference_df = pd.DataFrame(ref_data, index=dates[:100])
    
    # 当前数据（有漂移）
    cur_data = {
        'feature1': np.random.normal(0.5, 1, 50),  # 均值漂移
        'feature2': np.random.normal(10, 3, 50),   # 方差漂移
        'feature3': np.random.exponential(1.5, 50) # 参数漂移
    }
    current_df = pd.DataFrame(cur_data, index=dates[100:150])
    
    # 创建漂移检测器
    detector = DriftDetector()
    
    # 设置参考数据
    detector.set_reference_data(reference_df)
    
    # 检测分布漂移
    drift_result = detector.detect_distribution_drift(current_df, method='ks_test')
    print("分布漂移检测结果:", drift_result['drift_detected'])
    print("漂移特征:", drift_result['drift_features'])
    
    # 检测协变量漂移
    covariate_result = detector.detect_covariate_drift(current_df, method='pca')
    print("协变量漂移检测结果:", covariate_result['drift_detected'])
    print("漂移分数:", covariate_result['drift_score'])
    
    # 获取摘要
    summary = detector.get_drift_summary()
    print("漂移检测摘要:", summary) class FeatureDriftDetector:
    def __init__(self, reference_data):
        self.reference = reference_data
    
    def detect_drift(self, current_data, feature_name, threshold=0.05):
        """检测特征分布漂移"""
        ref_feature = self.reference[feature_name]
        curr_feature = current_data[feature_name]
        
        # KS检验检测分布变化
        statistic, p_value = ks_2samp(ref_feature, curr_feature)
        if p_value < threshold:
            return True, f"{feature_name} 检测到显著漂移 (p={p_value:.4f})"
        return False, ""
