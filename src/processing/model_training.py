#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习模型训练模块
提供模型训练、评估、预测、回测等功能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score, train_test_split
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, target_column: str = 'target', 
                 prediction_horizon: int = 1,
                 model_type: str = 'regression'):
        """
        初始化模型训练器
        
        Args:
            target_column: 目标变量列名
            prediction_horizon: 预测时间窗口
            model_type: 模型类型 ('regression', 'classification')
        """
        self.target_column = target_column
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def prepare_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备目标变量
        
        Args:
            df: 数据DataFrame
            
        Returns:
            包含目标变量的DataFrame
        """
        result = df.copy()
        
        if self.model_type == 'regression':
            # 回归任务：预测未来收益率
            if 'ClosePrice' in result.columns:
                result[self.target_column] = result['ClosePrice'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            elif 'Close' in result.columns:
                result[self.target_column] = result['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        elif self.model_type == 'classification':
            # 分类任务：预测涨跌
            if 'ClosePrice' in result.columns:
                returns = result['ClosePrice'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            elif 'Close' in result.columns:
                returns = result['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            
            result[self.target_column] = (returns > 0).astype(int)
        
        return result
    
    def train_linear_models(self, df: pd.DataFrame,
                           models: List[str] = ['linear', 'ridge', 'lasso']) -> Dict[str, Any]:
        """
        训练线性模型
        
        Args:
            df: 数据DataFrame
            models: 要训练的模型列表
            
        Returns:
            训练结果字典
        """
        logger.info("开始训练线性模型")
        
        # 准备数据
        df = self.prepare_target_variable(df)
        df = df.dropna()
        
        if len(df) == 0:
            logger.warning("数据为空，无法训练模型")
            return {}
        
        # 分离特征和目标
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # 移除非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        for model_name in models:
            try:
                if model_name == 'linear':
                    model = LinearRegression()
                elif model_name == 'ridge':
                    model = Ridge(alpha=1.0)
                elif model_name == 'lasso':
                    model = Lasso(alpha=0.01)
                else:
                    continue
                
                # 训练模型
                model.fit(X_scaled, y)
                
                # 预测
                y_pred = model.predict(X_scaled)
                
                # 评估模型
                metrics = self._evaluate_model(y, y_pred, model_name)
                
                # 保存模型和结果
                self.models[model_name] = model
                self.scalers[model_name] = scaler
                self.performance_metrics[model_name] = metrics
                
                # 特征重要性（系数）
                if hasattr(model, 'coef_'):
                    importance = dict(zip(X.columns, np.abs(model.coef_)))
                    self.feature_importance[model_name] = importance
                
                results[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'metrics': metrics,
                    'feature_importance': self.feature_importance.get(model_name, {})
                }
                
                logger.info(f"{model_name} 模型训练完成")
                
            except Exception as e:
                logger.error(f"{model_name} 模型训练失败: {e}")
        
        return results
    
    def train_ensemble_models(self, df: pd.DataFrame,
                             models: List[str] = ['random_forest', 'gradient_boosting']) -> Dict[str, Any]:
        """
        训练集成模型
        
        Args:
            df: 数据DataFrame
            models: 要训练的模型列表
            
        Returns:
            训练结果字典
        """
        logger.info("开始训练集成模型")
        
        # 准备数据
        df = self.prepare_target_variable(df)
        df = df.dropna()
        
        if len(df) == 0:
            logger.warning("数据为空，无法训练模型")
            return {}
        
        # 分离特征和目标
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # 移除非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        results = {}
        
        for model_name in models:
            try:
                if model_name == 'random_forest':
                    if self.model_type == 'regression':
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    else:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                elif model_name == 'gradient_boosting':
                    if self.model_type == 'regression':
                        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    else:
                        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                
                else:
                    continue
                
                # 训练模型
                model.fit(X, y)
                
                # 预测
                y_pred = model.predict(X)
                
                # 评估模型
                metrics = self._evaluate_model(y, y_pred, model_name)
                
                # 保存模型和结果
                self.models[model_name] = model
                self.performance_metrics[model_name] = metrics
                
                # 特征重要性
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(X.columns, model.feature_importances_))
                    self.feature_importance[model_name] = importance
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_importance': self.feature_importance.get(model_name, {})
                }
                
                logger.info(f"{model_name} 模型训练完成")
                
            except Exception as e:
                logger.error(f"{model_name} 模型训练失败: {e}")
        
        return results
    
    def _evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            
        Returns:
            性能指标字典
        """
        metrics = {}
        
        if self.model_type == 'regression':
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
            
        elif self.model_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            # 二分类特有指标
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['auc'] = roc_auc_score(y_true, y_pred)
                except:
                    metrics['auc'] = 0.5
        
        return metrics
    
    def cross_validate_model(self, df: pd.DataFrame, 
                           model_name: str = 'random_forest',
                           n_splits: int = 5) -> Dict[str, List[float]]:
        """
        交叉验证模型
        
        Args:
            df: 数据DataFrame
            model_name: 模型名称
            n_splits: 分割次数
            
        Returns:
            交叉验证结果
        """
        logger.info(f"开始交叉验证 {model_name} 模型")
        
        # 准备数据
        df = self.prepare_target_variable(df)
        df = df.dropna()
        
        if len(df) == 0:
            logger.warning("数据为空，无法进行交叉验证")
            return {}
        
        # 分离特征和目标
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # 移除非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # 选择模型
        if model_name == 'random_forest':
            if self.model_type == 'regression':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        elif model_name == 'gradient_boosting':
            if self.model_type == 'regression':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        else:
            logger.error(f"不支持的模型类型: {model_name}")
            return {}
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'train_scores': [],
            'test_scores': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 评估
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            cv_scores['train_scores'].append(train_score)
            cv_scores['test_scores'].append(test_score)
        
        logger.info(f"交叉验证完成，平均测试分数: {np.mean(cv_scores['test_scores']):.4f}")
        return cv_scores
    
    def predict(self, df: pd.DataFrame, model_name: str = 'random_forest') -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Args:
            df: 数据DataFrame
            model_name: 模型名称
            
        Returns:
            预测结果
        """
        if model_name not in self.models:
            logger.error(f"模型 {model_name} 未训练")
            return np.array([])
        
        model = self.models[model_name]
        
        # 准备特征
        X = df.copy()
        if self.target_column in X.columns:
            X = X.drop(columns=[self.target_column])
        
        # 移除非数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # 标准化（如果需要）
        if model_name in self.scalers:
            X = self.scalers[model_name].transform(X)
        
        # 预测
        predictions = model.predict(X)
        
        return predictions
    
    def backtest_strategy(self, df: pd.DataFrame,
                         model_name: str = 'random_forest',
                         initial_capital: float = 100000,
                         transaction_cost: float = 0.001) -> Dict[str, Any]:
        """
        回测交易策略
        
        Args:
            df: 数据DataFrame
            model_name: 模型名称
            initial_capital: 初始资金
            transaction_cost: 交易成本
            
        Returns:
            回测结果
        """
        logger.info(f"开始回测 {model_name} 策略")
        
        # 准备数据
        df = self.prepare_target_variable(df)
        df = df.dropna()
        
        if len(df) == 0:
            logger.warning("数据为空，无法进行回测")
            return {}
        
        # 获取预测
        predictions = self.predict(df, model_name)
        
        if len(predictions) == 0:
            return {}
        
        # 计算策略收益
        if self.model_type == 'regression':
            # 回归模型：直接使用预测的收益率
            strategy_returns = predictions
        else:
            # 分类模型：根据预测信号计算收益
            if 'ClosePrice' in df.columns:
                actual_returns = df['ClosePrice'].pct_change()
            elif 'Close' in df.columns:
                actual_returns = df['Close'].pct_change()
            else:
                logger.error("无法找到价格数据")
                return {}
            
            # 根据预测信号调整收益
            strategy_returns = actual_returns * (predictions * 2 - 1)  # 转换为 -1, 1
        
        # 计算累积收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 计算策略指标
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (strategy_returns > 0).mean()
        
        backtest_results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'cumulative_returns': cumulative_returns,
            'strategy_returns': strategy_returns,
            'predictions': predictions
        }
        
        logger.info(f"回测完成，年化收益率: {annual_return:.2%}, 夏普比率: {sharpe_ratio:.2f}")
        return backtest_results
    
    def save_model(self, model_name: str, filepath: str):
        """保存模型"""
        if model_name not in self.models:
            logger.error(f"模型 {model_name} 不存在")
            return
        
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scalers.get(model_name),
            'feature_importance': self.feature_importance.get(model_name, {}),
            'performance_metrics': self.performance_metrics.get(model_name, {}),
            'model_type': self.model_type,
            'target_column': self.target_column,
            'prediction_horizon': self.prediction_horizon
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> str:
        """加载模型"""
        try:
            model_data = joblib.load(filepath)
            
            model_name = 'loaded_model'
            self.models[model_name] = model_data['model']
            if model_data['scaler']:
                self.scalers[model_name] = model_data['scaler']
            self.feature_importance[model_name] = model_data['feature_importance']
            self.performance_metrics[model_name] = model_data['performance_metrics']
            
            logger.info(f"模型已从 {filepath} 加载")
            return model_name
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return ""
    
    def get_model_summary(self) -> pd.DataFrame:
        """获取模型汇总报告"""
        summary_data = []
        
        for model_name, metrics in self.performance_metrics.items():
            summary_data.append({
                'model': model_name,
                **metrics
            })
        
        return pd.DataFrame(summary_data) 