#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征筛选模块
提供特征选择、重要性评估、相关性分析、稳定性检验等功能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, target_column: str = 'target'):
        """
        初始化特征选择器
        
        Args:
            target_column: 目标变量列名
        """
        self.target_column = target_column
        self.selected_features = []
        self.feature_importance = {}
        self.correlation_matrix = None
        self.stability_scores = {}
        
    def filter_by_correlation(self, df: pd.DataFrame, 
                            threshold: float = 0.95,
                            method: str = 'pearson') -> pd.DataFrame:
        """
        基于相关性过滤特征
        
        Args:
            df: 数据DataFrame
            threshold: 相关性阈值
            method: 相关性计算方法 ('pearson', 'spearman', 'kendall')
            
        Returns:
            过滤后的DataFrame
        """
        logger.info(f"开始基于相关性过滤特征，阈值: {threshold}")
        
        # 计算相关性矩阵
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.correlation_matrix = df[numeric_cols].corr(method=method)
        
        # 找出高相关性的特征对
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = abs(self.correlation_matrix.iloc[i, j])
                if corr_value > threshold:
                    high_corr_pairs.append((
                        self.correlation_matrix.columns[i],
                        self.correlation_matrix.columns[j],
                        corr_value
                    ))
        
        # 移除高相关性特征
        features_to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            # 保留与目标变量相关性更高的特征
            if self.target_column in df.columns:
                corr1 = abs(df[feat1].corr(df[self.target_column]))
                corr2 = abs(df[feat2].corr(df[self.target_column]))
                if corr1 < corr2:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
            else:
                # 如果没有目标变量，保留方差更大的特征
                var1 = df[feat1].var()
                var2 = df[feat2].var()
                if var1 < var2:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
        
        # 过滤特征
        result = df.drop(columns=list(features_to_remove))
        
        logger.info(f"相关性过滤完成，移除 {len(features_to_remove)} 个特征")
        return result
    
    def filter_by_variance(self, df: pd.DataFrame, 
                          threshold: float = 0.01) -> pd.DataFrame:
        """
        基于方差过滤特征
        
        Args:
            df: 数据DataFrame
            threshold: 方差阈值
            
        Returns:
            过滤后的DataFrame
        """
        logger.info(f"开始基于方差过滤特征，阈值: {threshold}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numeric_cols].var()
        
        # 找出低方差特征
        low_var_features = variances[variances < threshold].index.tolist()
        
        # 过滤特征
        result = df.drop(columns=low_var_features)
        
        logger.info(f"方差过滤完成，移除 {len(low_var_features)} 个特征")
        return result
    
    def filter_by_missing_rate(self, df: pd.DataFrame, 
                              threshold: float = 0.5) -> pd.DataFrame:
        """
        基于缺失率过滤特征
        
        Args:
            df: 数据DataFrame
            threshold: 缺失率阈值
            
        Returns:
            过滤后的DataFrame
        """
        logger.info(f"开始基于缺失率过滤特征，阈值: {threshold}")
        
        missing_rates = df.isnull().sum() / len(df)
        high_missing_features = missing_rates[missing_rates > threshold].index.tolist()
        
        # 过滤特征
        result = df.drop(columns=high_missing_features)
        
        logger.info(f"缺失率过滤完成，移除 {len(high_missing_features)} 个特征")
        return result
    
    def select_by_statistical_tests(self, df: pd.DataFrame, 
                                   k: int = 100,
                                   score_func: str = 'f_regression') -> pd.DataFrame:
        """
        基于统计检验选择特征
        
        Args:
            df: 数据DataFrame
            k: 选择的特征数量
            score_func: 评分函数 ('f_regression', 'mutual_info_regression')
            
        Returns:
            选择后的DataFrame
        """
        logger.info(f"开始基于统计检验选择特征，选择数量: {k}")
        
        if self.target_column not in df.columns:
            logger.warning("目标变量不存在，跳过统计检验选择")
            return df
        
        # 准备数据
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # 移除非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # 处理缺失值
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # 选择评分函数
        if score_func == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
        elif score_func == 'mutual_info_regression':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(X.columns)))
        else:
            raise ValueError(f"不支持的评分函数: {score_func}")
        
        # 拟合选择器
        selector.fit(X, y)
        
        # 获取选择的特征
        selected_features = X.columns[selector.get_support()].tolist()
        
        # 保存特征重要性
        scores = selector.scores_
        for i, feature in enumerate(X.columns):
            self.feature_importance[feature] = scores[i]
        
        # 选择特征
        result = df[selected_features + [self.target_column] if self.target_column in df.columns else selected_features]
        
        logger.info(f"统计检验选择完成，选择 {len(selected_features)} 个特征")
        return result
    
    def select_by_model_importance(self, df: pd.DataFrame,
                                  k: int = 100,
                                  model_type: str = 'random_forest') -> pd.DataFrame:
        """
        基于模型重要性选择特征
        
        Args:
            df: 数据DataFrame
            k: 选择的特征数量
            model_type: 模型类型 ('random_forest', 'lasso', 'ridge')
            
        Returns:
            选择后的DataFrame
        """
        logger.info(f"开始基于模型重要性选择特征，模型: {model_type}")
        
        if self.target_column not in df.columns:
            logger.warning("目标变量不存在，跳过模型重要性选择")
            return df
        
        # 准备数据
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # 移除非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # 处理缺失值
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 选择模型
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01, random_state=42)
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 使用RFE进行特征选择
        selector = RFE(estimator=model, n_features_to_select=min(k, len(X.columns)))
        selector.fit(X_scaled, y)
        
        # 获取选择的特征
        selected_features = X.columns[selector.get_support()].tolist()
        
        # 保存特征重要性
        if hasattr(selector.estimator_, 'feature_importances_'):
            importances = selector.estimator_.feature_importances_
        elif hasattr(selector.estimator_, 'coef_'):
            importances = np.abs(selector.estimator_.coef_)
        else:
            importances = np.ones(len(X.columns))
        
        for i, feature in enumerate(X.columns):
            self.feature_importance[feature] = importances[i]
        
        # 选择特征
        result = df[selected_features + [self.target_column] if self.target_column in df.columns else selected_features]
        
        logger.info(f"模型重要性选择完成，选择 {len(selected_features)} 个特征")
        return result
    
    def evaluate_feature_stability(self, df: pd.DataFrame,
                                 n_splits: int = 5,
                                 test_size: float = 0.3) -> Dict[str, float]:
        """
        评估特征稳定性
        
        Args:
            df: 数据DataFrame
            n_splits: 分割次数
            test_size: 测试集比例
            
        Returns:
            特征稳定性分数字典
        """
        logger.info("开始评估特征稳定性")
        
        if self.target_column not in df.columns:
            logger.warning("目标变量不存在，跳过稳定性评估")
            return {}
        
        from sklearn.model_selection import train_test_split
        
        # 准备数据
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # 移除非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # 处理缺失值
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        stability_scores = {}
        
        for feature in X.columns:
            feature_stability = []
            
            for _ in range(n_splits):
                # 分割数据
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=np.random.randint(1000)
                )
                
                # 计算特征重要性
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # 获取特征重要性
                feature_idx = list(X.columns).index(feature)
                importance = model.feature_importances_[feature_idx]
                feature_stability.append(importance)
            
            # 计算稳定性分数（变异系数的倒数）
            mean_importance = np.mean(feature_stability)
            std_importance = np.std(feature_stability)
            
            if mean_importance > 0:
                stability_scores[feature] = mean_importance / (std_importance + 1e-8)
            else:
                stability_scores[feature] = 0
        
        self.stability_scores = stability_scores
        
        logger.info("特征稳定性评估完成")
        return stability_scores
    
    def filter_by_stability(self, df: pd.DataFrame,
                           threshold: float = 0.1) -> pd.DataFrame:
        """
        基于稳定性过滤特征
        
        Args:
            df: 数据DataFrame
            threshold: 稳定性阈值
            
        Returns:
            过滤后的DataFrame
        """
        logger.info(f"开始基于稳定性过滤特征，阈值: {threshold}")
        
        # 评估稳定性
        stability_scores = self.evaluate_feature_stability(df)
        
        # 过滤低稳定性特征
        low_stability_features = [
            feature for feature, score in stability_scores.items()
            if score < threshold
        ]
        
        # 过滤特征
        result = df.drop(columns=low_stability_features)
        
        logger.info(f"稳定性过滤完成，移除 {len(low_stability_features)} 个特征")
        return result
    
    def comprehensive_feature_selection(self, df: pd.DataFrame,
                                      correlation_threshold: float = 0.95,
                                      variance_threshold: float = 0.01,
                                      missing_threshold: float = 0.5,
                                      stability_threshold: float = 0.1,
                                      final_k: int = 100) -> pd.DataFrame:
        """
        综合特征选择
        
        Args:
            df: 数据DataFrame
            correlation_threshold: 相关性阈值
            variance_threshold: 方差阈值
            missing_threshold: 缺失率阈值
            stability_threshold: 稳定性阈值
            final_k: 最终选择的特征数量
            
        Returns:
            选择后的DataFrame
        """
        logger.info("开始综合特征选择")
        
        result = df.copy()
        
        # 1. 基于缺失率过滤
        result = self.filter_by_missing_rate(result, missing_threshold)
        
        # 2. 基于方差过滤
        result = self.filter_by_variance(result, variance_threshold)
        
        # 3. 基于相关性过滤
        result = self.filter_by_correlation(result, correlation_threshold)
        
        # 4. 基于稳定性过滤（如果有目标变量）
        if self.target_column in result.columns:
            result = self.filter_by_stability(result, stability_threshold)
        
        # 5. 基于统计检验选择
        if self.target_column in result.columns:
            result = self.select_by_statistical_tests(result, final_k)
        else:
            # 如果没有目标变量，选择方差最大的特征
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            variances = result[numeric_cols].var()
            top_features = variances.nlargest(final_k).index.tolist()
            result = result[top_features]
        
        self.selected_features = [col for col in result.columns if col != self.target_column]
        
        logger.info(f"综合特征选择完成，最终选择 {len(self.selected_features)} 个特征")
        return result
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """获取特征重要性报告"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        # 创建报告
        report_data = []
        for feature, importance in self.feature_importance.items():
            stability = self.stability_scores.get(feature, 0)
            report_data.append({
                'feature': feature,
                'importance': importance,
                'stability': stability,
                'selected': feature in self.selected_features
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('importance', ascending=False)
        
        return report_df
    
    def get_correlation_report(self) -> pd.DataFrame:
        """获取相关性报告"""
        if self.correlation_matrix is None:
            return pd.DataFrame()
        
        # 找出高相关性特征对
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = self.correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # 高相关性阈值
                    high_corr_pairs.append({
                        'feature1': self.correlation_matrix.columns[i],
                        'feature2': self.correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False) 