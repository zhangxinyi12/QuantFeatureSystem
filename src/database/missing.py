"""
缺失值处理模块
提供多种缺失值填充策略和检测方法
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
import warnings

logger = logging.getLogger(__name__)

class MissingValueHandler:
    """缺失值处理器"""
    
    def __init__(self, strategy: str = 'auto'):
        """
        初始化缺失值处理器
        
        Args:
            strategy: 处理策略 ('auto', 'simple', 'knn', 'rf', 'interpolation')
        """
        self.strategy = strategy
        self.imputers = {}
        self.missing_patterns = {}
        
    def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析缺失值模式
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict: 缺失值分析结果
        """
        logger.info("开始分析缺失值模式")
        
        # 基本统计
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # 缺失值模式
        missing_patterns = {
            'total_missing': missing_counts.sum(),
            'total_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'columns_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'missing_patterns': {}
        }
        
        # 分析缺失值的相关性
        missing_matrix = df.isnull()
        if missing_matrix.sum().sum() > 0:
            # 计算缺失值之间的相关性
            missing_corr = missing_matrix.corr()
            missing_patterns['missing_correlations'] = missing_corr.to_dict()
            
            # 识别缺失值模式
            pattern_counts = missing_matrix.value_counts()
            missing_patterns['pattern_counts'] = pattern_counts.head(10).to_dict()
        
        self.missing_patterns = missing_patterns
        logger.info(f"缺失值分析完成，总缺失值: {missing_patterns['total_missing']}")
        
        return missing_patterns
    
    def detect_missing_patterns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        检测缺失值模式类型
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict: 每列的缺失值模式类型
        """
        patterns = {}
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                missing_indices = df[col].isnull()
                
                # 检查是否为完全随机缺失(MCAR)
                if self._is_mcar(df, col):
                    patterns[col] = 'MCAR'  # Missing Completely At Random
                # 检查是否为随机缺失(MAR)
                elif self._is_mar(df, col):
                    patterns[col] = 'MAR'   # Missing At Random
                # 否则为非随机缺失(MNAR)
                else:
                    patterns[col] = 'MNAR'  # Missing Not At Random
            else:
                patterns[col] = 'No Missing'
        
        return patterns
    
    def _is_mcar(self, df: pd.DataFrame, col: str) -> bool:
        """检查是否为完全随机缺失"""
        # 简单的MCAR检测：检查缺失值是否与时间或其他变量无关
        missing_indices = df[col].isnull()
        
        # 检查与时间的关系（如果有时间列）
        time_columns = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if time_columns:
            time_col = time_columns[0]
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                # 检查缺失值是否在时间上随机分布
                time_series = df[time_col].dt.dayofweek
                missing_by_day = time_series[missing_indices].value_counts()
                all_by_day = time_series.value_counts()
                
                # 计算卡方统计量
                expected = all_by_day * missing_indices.sum() / len(df)
                chi_square = ((missing_by_day - expected) ** 2 / expected).sum()
                
                # 如果卡方值较小，认为是MCAR
                return chi_square < 12.59  # 5%显著性水平，6个自由度
        
        return True  # 默认假设为MCAR
    
    def _is_mar(self, df: pd.DataFrame, col: str) -> bool:
        """检查是否为随机缺失"""
        # 检查缺失值是否与其他变量相关
        missing_indices = df[col].isnull()
        
        # 检查与其他数值变量的相关性
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != col]
        
        correlations = []
        for other_col in numeric_cols:
            if other_col in df.columns and not df[other_col].isnull().all():
                # 计算缺失值指示符与其他变量的相关性
                corr = missing_indices.astype(int).corr(df[other_col])
                if not pd.isna(corr):
                    correlations.append(abs(corr))
        
        # 如果与任何变量的相关性都很低，认为是MAR
        return len(correlations) == 0 or max(correlations) < 0.3
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Optional[str] = None,
                            **kwargs) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 数据DataFrame
            strategy: 处理策略
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        strategy = strategy or self.strategy
        
        if strategy == 'auto':
            return self._auto_handle_missing(df, **kwargs)
        elif strategy == 'simple':
            return self._simple_imputation(df, **kwargs)
        elif strategy == 'knn':
            return self._knn_imputation(df, **kwargs)
        elif strategy == 'rf':
            return self._random_forest_imputation(df, **kwargs)
        elif strategy == 'interpolation':
            return self._interpolation_imputation(df, **kwargs)
        else:
            raise ValueError(f"不支持的策略: {strategy}")
    
    def _auto_handle_missing(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """自动选择缺失值处理策略"""
        logger.info("使用自动策略处理缺失值")
        
        # 分析缺失值模式
        patterns = self.analyze_missing_patterns(df)
        
        # 根据缺失值比例选择策略
        total_missing_pct = patterns['total_percentage']
        
        if total_missing_pct < 5:
            # 缺失值很少，使用简单填充
            return self._simple_imputation(df, **kwargs)
        elif total_missing_pct < 20:
            # 缺失值中等，使用插值
            return self._interpolation_imputation(df, **kwargs)
        else:
            # 缺失值较多，使用高级方法
            return self._knn_imputation(df, **kwargs)
    
    def _simple_imputation(self, df: pd.DataFrame, 
                          method: str = 'mean',
                          **kwargs) -> pd.DataFrame:
        """
        简单填充方法
        
        Args:
            df: 数据DataFrame
            method: 填充方法 ('mean', 'median', 'mode', 'constant')
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        logger.info(f"使用简单填充方法: {method}")
        
        result_df = df.copy()
        
        # 按数据类型分别处理
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns
        
        # 处理数值列
        if len(numeric_cols) > 0:
            if method == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif method == 'median':
                imputer = SimpleImputer(strategy='median')
            elif method == 'constant':
                constant_value = kwargs.get('constant_value', 0)
                imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
            else:
                raise ValueError(f"不支持的数值填充方法: {method}")
            
            result_df[numeric_cols] = imputer.fit_transform(result_df[numeric_cols])
            self.imputers['numeric'] = imputer
        
        # 处理分类列
        if len(categorical_cols) > 0:
            if method == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            elif method == 'constant':
                constant_value = kwargs.get('constant_value', 'Unknown')
                imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
            else:
                # 默认使用众数
                imputer = SimpleImputer(strategy='most_frequent')
            
            result_df[categorical_cols] = imputer.fit_transform(result_df[categorical_cols])
            self.imputers['categorical'] = imputer
        
        logger.info("简单填充完成")
        return result_df
    
    def _knn_imputation(self, df: pd.DataFrame, 
                       n_neighbors: int = 5,
                       **kwargs) -> pd.DataFrame:
        """
        KNN填充方法
        
        Args:
            df: 数据DataFrame
            n_neighbors: 邻居数量
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        logger.info(f"使用KNN填充方法，邻居数: {n_neighbors}")
        
        # 只处理数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("没有数值列，回退到简单填充")
            return self._simple_imputation(df, **kwargs)
        
        # 创建KNN填充器
        imputer = KNNImputer(n_neighbors=n_neighbors)
        
        # 填充数值列
        result_df = df.copy()
        result_df[numeric_cols] = imputer.fit_transform(result_df[numeric_cols])
        
        # 处理分类列（使用简单填充）
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            result_df[categorical_cols] = cat_imputer.fit_transform(result_df[categorical_cols])
        
        self.imputers['knn'] = imputer
        logger.info("KNN填充完成")
        return result_df
    
    def _random_forest_imputation(self, df: pd.DataFrame, 
                                n_estimators: int = 100,
                                **kwargs) -> pd.DataFrame:
        """
        随机森林填充方法
        
        Args:
            df: 数据DataFrame
            n_estimators: 树的数量
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        logger.info(f"使用随机森林填充方法，树的数量: {n_estimators}")
        
        # 只处理数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("没有数值列，回退到简单填充")
            return self._simple_imputation(df, **kwargs)
        
        result_df = df.copy()
        
        # 为每个有缺失值的列训练一个随机森林模型
        for col in numeric_cols:
            if result_df[col].isnull().sum() > 0:
                # 准备训练数据
                train_data = result_df.dropna(subset=[col])
                if len(train_data) == 0:
                    continue
                
                # 选择特征列（排除目标列和其他有缺失值的列）
                feature_cols = [c for c in numeric_cols if c != col and not result_df[c].isnull().all()]
                
                if len(feature_cols) == 0:
                    continue
                
                # 训练模型
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                model.fit(train_data[feature_cols], train_data[col])
                
                # 预测缺失值
                missing_mask = result_df[col].isnull()
                if missing_mask.sum() > 0:
                    predictions = model.predict(result_df.loc[missing_mask, feature_cols])
                    result_df.loc[missing_mask, col] = predictions
        
        # 处理分类列
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            result_df[categorical_cols] = cat_imputer.fit_transform(result_df[categorical_cols])
        
        logger.info("随机森林填充完成")
        return result_df
    
    def _interpolation_imputation(self, df: pd.DataFrame, 
                                method: str = 'linear',
                                **kwargs) -> pd.DataFrame:
        """
        插值填充方法
        
        Args:
            df: 数据DataFrame
            method: 插值方法 ('linear', 'polynomial', 'spline', 'time')
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        logger.info(f"使用插值填充方法: {method}")
        
        result_df = df.copy()
        
        # 按股票代码分组进行插值
        if 'SecuCode' in result_df.columns:
            for secu_code in result_df['SecuCode'].unique():
                mask = result_df['SecuCode'] == secu_code
                stock_data = result_df[mask].copy()
                
                # 按时间排序
                if 'TradingDay' in stock_data.columns:
                    stock_data = stock_data.sort_values('TradingDay')
                
                # 对数值列进行插值
                numeric_cols = stock_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if stock_data[col].isnull().sum() > 0:
                        if method == 'linear':
                            stock_data[col] = stock_data[col].interpolate(method='linear')
                        elif method == 'polynomial':
                            stock_data[col] = stock_data[col].interpolate(method='polynomial', order=2)
                        elif method == 'spline':
                            stock_data[col] = stock_data[col].interpolate(method='spline', order=3)
                        elif method == 'time':
                            stock_data[col] = stock_data[col].interpolate(method='time')
                        else:
                            stock_data[col] = stock_data[col].interpolate(method='linear')
                        
                        # 处理边界缺失值
                        stock_data[col] = stock_data[col].fillna(method='ffill').fillna(method='bfill')
                
                result_df.loc[mask] = stock_data
        else:
            # 没有分组列，直接插值
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].interpolate(method=method)
                    result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
        
        # 处理分类列
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
        
        logger.info("插值填充完成")
        return result_df
    
    def get_imputation_report(self, original_df: pd.DataFrame, 
                            imputed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成填充报告
        
        Args:
            original_df: 原始数据
            imputed_df: 填充后的数据
            
        Returns:
            Dict: 填充报告
        """
        report = {
            'original_missing': original_df.isnull().sum().to_dict(),
            'imputed_missing': imputed_df.isnull().sum().to_dict(),
            'imputation_summary': {},
            'data_quality_metrics': {}
        }
        
        # 填充摘要
        for col in original_df.columns:
            original_missing = original_df[col].isnull().sum()
            imputed_missing = imputed_df[col].isnull().sum()
            
            if original_missing > 0:
                report['imputation_summary'][col] = {
                    'original_missing': original_missing,
                    'imputed_missing': imputed_missing,
                    'filled_count': original_missing - imputed_missing,
                    'fill_rate': (original_missing - imputed_missing) / original_missing * 100
                }
        
        # 数据质量指标
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in original_df.columns and col in imputed_df.columns:
                original_stats = original_df[col].describe()
                imputed_stats = imputed_df[col].describe()
                
                report['data_quality_metrics'][col] = {
                    'original_mean': original_stats['mean'],
                    'imputed_mean': imputed_stats['mean'],
                    'original_std': original_stats['std'],
                    'imputed_std': imputed_stats['std'],
                    'mean_change': imputed_stats['mean'] - original_stats['mean'],
                    'std_change': imputed_stats['std'] - original_stats['std']
                }
        
        return report
