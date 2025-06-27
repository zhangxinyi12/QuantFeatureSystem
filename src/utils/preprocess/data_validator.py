#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证工具模块
提供数据质量检查、前瞻性偏差检测、数据完整性验证等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import warnings

class DataValidator:
    """数据验证器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化数据验证器
        
        参数:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.validation_results = {}
        
    def validate_dataframe(self, df: pd.DataFrame, 
                          required_columns: List[str] = None,
                          date_column: str = None,
                          check_duplicates: bool = True,
                          check_missing: bool = True,
                          check_data_types: bool = True) -> Dict:
        """
        验证DataFrame的基本质量
        
        参数:
            df: 要验证的DataFrame
            required_columns: 必需的列名列表
            date_column: 日期列名
            check_duplicates: 是否检查重复值
            check_missing: 是否检查缺失值
            check_data_types: 是否检查数据类型
            
        返回:
            验证结果字典
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # 检查DataFrame是否为空
        if df.empty:
            results['valid'] = False
            results['errors'].append("DataFrame为空")
            return results
            
        # 检查必需列
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                results['valid'] = False
                results['errors'].append(f"缺少必需列: {missing_columns}")
                
        # 检查重复值
        if check_duplicates:
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                results['warnings'].append(f"发现 {duplicate_count} 行重复数据")
                results['summary']['duplicate_rows'] = duplicate_count
                
        # 检查缺失值
        if check_missing:
            missing_info = self._check_missing_values(df)
            if missing_info['total_missing'] > 0:
                results['warnings'].append(f"发现缺失值: {missing_info['total_missing']} 个")
                results['summary']['missing_values'] = missing_info
                
        # 检查数据类型
        if check_data_types:
            type_info = self._check_data_types(df)
            results['summary']['data_types'] = type_info
            
        # 检查日期列
        if date_column and date_column in df.columns:
            date_validation = self._validate_date_column(df[date_column])
            if not date_validation['valid']:
                results['warnings'].extend(date_validation['issues'])
                
        # 更新验证状态
        if results['errors']:
            results['valid'] = False
            
        self.validation_results['dataframe_validation'] = results
        return results
        
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """检查缺失值"""
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        return {
            'total_missing': total_missing,
            'missing_by_column': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentage': (total_missing / (df.shape[0] * df.shape[1])) * 100
        }
        
    def _check_data_types(self, df: pd.DataFrame) -> Dict:
        """检查数据类型"""
        return {
            'dtypes': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
            'object_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
    def _validate_date_column(self, date_series: pd.Series) -> Dict:
        """验证日期列"""
        issues = []
        
        # 检查是否为datetime类型
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            issues.append("日期列不是datetime类型")
            
        # 检查日期范围
        if not date_series.empty:
            min_date = date_series.min()
            max_date = date_series.max()
            current_date = pd.Timestamp.now()
            
            if max_date > current_date:
                issues.append(f"发现未来日期: {max_date}")
                
            if min_date < pd.Timestamp('1900-01-01'):
                issues.append(f"发现异常早期日期: {min_date}")
                
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
        
    def check_lookahead_bias(self, df: pd.DataFrame, 
                            date_column: str,
                            feature_columns: List[str],
                            target_column: str = None,
                            lookback_days: int = 1) -> Dict:
        """
        检查前瞻性偏差
        
        参数:
            df: 数据DataFrame
            date_column: 日期列名
            feature_columns: 特征列名列表
            target_column: 目标列名（可选）
            lookback_days: 回看天数
            
        返回:
            前瞻性偏差检查结果
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'lookahead_issues': []
        }
        
        if date_column not in df.columns:
            results['valid'] = False
            results['errors'].append(f"日期列 {date_column} 不存在")
            return results
            
        # 确保日期列是datetime类型
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # 按日期排序
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # 检查每个特征列
        for feature in feature_columns:
            if feature not in df.columns:
                results['warnings'].append(f"特征列 {feature} 不存在")
                continue
                
            # 检查前瞻性偏差
            lookahead_issues = self._check_feature_lookahead_bias(
                df, date_column, feature, target_column, lookback_days
            )
            
            if lookahead_issues:
                results['lookahead_issues'].extend(lookahead_issues)
                results['valid'] = False
                
        # 检查目标列的前瞻性偏差
        if target_column and target_column in df.columns:
            target_issues = self._check_target_lookahead_bias(
                df, date_column, target_column
            )
            if target_issues:
                results['lookahead_issues'].extend(target_issues)
                results['valid'] = False
                
        self.validation_results['lookahead_bias'] = results
        return results
        
    def _check_feature_lookahead_bias(self, df: pd.DataFrame, 
                                     date_column: str,
                                     feature_column: str,
                                     target_column: str = None,
                                     lookback_days: int = 1) -> List[str]:
        """检查单个特征的前瞻性偏差"""
        issues = []
        
        # 计算特征的变化率
        df[f'{feature_column}_change'] = df[feature_column].pct_change()
        
        # 检查是否存在异常大的变化
        change_threshold = 0.5  # 50%的变化阈值
        large_changes = df[abs(df[f'{feature_column}_change']) > change_threshold]
        
        if not large_changes.empty:
            for idx, row in large_changes.iterrows():
                change_date = row[date_column]
                change_value = row[f'{feature_column}_change']
                
                # 检查变化后的目标值（如果存在）
                if target_column and target_column in df.columns:
                    if idx < len(df) - 1:
                        next_target = df.iloc[idx + 1][target_column]
                        current_target = row[target_column]
                        
                        # 如果特征变化与目标变化方向一致，可能存在前瞻性偏差
                        if (change_value > 0 and next_target > current_target) or \
                           (change_value < 0 and next_target < current_target):
                            issues.append(
                                f"特征 {feature_column} 在 {change_date} 发生 {change_value:.2%} 变化，"
                                f"可能与目标值前瞻性相关"
                            )
                else:
                    issues.append(
                        f"特征 {feature_column} 在 {change_date} 发生异常变化: {change_value:.2%}"
                    )
                    
        return issues
        
    def _check_target_lookahead_bias(self, df: pd.DataFrame, 
                                    date_column: str,
                                    target_column: str) -> List[str]:
        """检查目标列的前瞻性偏差"""
        issues = []
        
        # 检查目标值是否使用了未来信息
        # 这里主要检查目标值的计算逻辑是否正确
        
        # 检查目标值的变化是否过于平滑（可能使用了未来信息）
        target_changes = df[target_column].pct_change()
        target_volatility = target_changes.std()
        
        if target_volatility < 0.001:  # 异常低的波动率
            issues.append(f"目标列 {target_column} 波动率异常低 ({target_volatility:.6f})，"
                         f"可能存在前瞻性偏差")
            
        return issues
        
    def check_data_consistency(self, df: pd.DataFrame, 
                              numeric_columns: List[str] = None,
                              categorical_columns: List[str] = None) -> Dict:
        """
        检查数据一致性
        
        参数:
            df: 数据DataFrame
            numeric_columns: 数值列列表
            categorical_columns: 分类列列表
            
        返回:
            一致性检查结果
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'consistency_issues': []
        }
        
        # 检查数值列的一致性
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    numeric_issues = self._check_numeric_consistency(df[col], col)
                    results['consistency_issues'].extend(numeric_issues)
                    
        # 检查分类列的一致性
        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    categorical_issues = self._check_categorical_consistency(df[col], col)
                    results['consistency_issues'].extend(categorical_issues)
                    
        # 检查跨列的一致性
        cross_column_issues = self._check_cross_column_consistency(df)
        results['consistency_issues'].extend(cross_column_issues)
        
        if results['consistency_issues']:
            results['valid'] = False
            
        self.validation_results['data_consistency'] = results
        return results
        
    def _check_numeric_consistency(self, series: pd.Series, column_name: str) -> List[str]:
        """检查数值列的一致性"""
        issues = []
        
        # 检查异常值
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        if len(outliers) > 0:
            issues.append(f"列 {column_name} 发现 {len(outliers)} 个异常值")
            
        # 检查负值（如果不应该有负值）
        if series.min() < 0 and column_name.lower() in ['price', 'volume', 'amount']:
            issues.append(f"列 {column_name} 包含负值，可能数据有误")
            
        # 检查零值
        zero_count = (series == 0).sum()
        if zero_count > len(series) * 0.5:  # 超过50%为零
            issues.append(f"列 {column_name} 超过50%的值为零")
            
        return issues
        
    def _check_categorical_consistency(self, series: pd.Series, column_name: str) -> List[str]:
        """检查分类列的一致性"""
        issues = []
        
        # 检查类别数量
        unique_count = series.nunique()
        if unique_count > len(series) * 0.8:  # 超过80%都是唯一值
            issues.append(f"列 {column_name} 类别过多，可能不是分类变量")
            
        # 检查类别分布
        value_counts = series.value_counts()
        if len(value_counts) > 0:
            max_freq = value_counts.max()
            min_freq = value_counts.min()
            if max_freq / min_freq > 100:  # 最大频率是最小频率的100倍以上
                issues.append(f"列 {column_name} 类别分布极不均匀")
                
        return issues
        
    def _check_cross_column_consistency(self, df: pd.DataFrame) -> List[str]:
        """检查跨列的一致性"""
        issues = []
        
        # 检查价格相关列的一致性
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower()]
        if len(price_columns) >= 2:
            for i, col1 in enumerate(price_columns):
                for col2 in price_columns[i+1:]:
                    if col1 in df.columns and col2 in df.columns:
                        # 检查价格逻辑关系
                        if 'high' in col1.lower() and 'low' in col2.lower():
                            invalid_rows = df[df[col1] < df[col2]]
                            if len(invalid_rows) > 0:
                                issues.append(f"发现 {len(invalid_rows)} 行数据中 {col1} < {col2}")
                                
        # 检查成交量相关列的一致性
        volume_columns = [col for col in df.columns if 'volume' in col.lower()]
        if len(volume_columns) >= 2:
            for i, col1 in enumerate(volume_columns):
                for col2 in volume_columns[i+1:]:
                    if col1 in df.columns and col2 in df.columns:
                        # 检查成交量逻辑关系
                        if 'total' in col1.lower() and 'partial' in col2.lower():
                            invalid_rows = df[df[col1] < df[col2]]
                            if len(invalid_rows) > 0:
                                issues.append(f"发现 {len(invalid_rows)} 行数据中 {col1} < {col2}")
                                
        return issues
        
    def generate_validation_report(self) -> str:
        """生成验证报告"""
        if not self.validation_results:
            return "没有验证结果"
            
        report = "数据验证报告\n"
        report += "=" * 50 + "\n\n"
        
        for validation_type, results in self.validation_results.items():
            report += f"验证类型: {validation_type}\n"
            report += f"验证结果: {'通过' if results['valid'] else '失败'}\n"
            
            if results['errors']:
                report += "错误:\n"
                for error in results['errors']:
                    report += f"  - {error}\n"
                    
            if results['warnings']:
                report += "警告:\n"
                for warning in results['warnings']:
                    report += f"  - {warning}\n"
                    
            if 'lookahead_issues' in results and results['lookahead_issues']:
                report += "前瞻性偏差问题:\n"
                for issue in results['lookahead_issues']:
                    report += f"  - {issue}\n"
                    
            if 'consistency_issues' in results and results['consistency_issues']:
                report += "一致性问题:\n"
                for issue in results['consistency_issues']:
                    report += f"  - {issue}\n"
                    
            report += "\n"
            
        return report

# 便捷函数
def quick_validate(df: pd.DataFrame, 
                   required_columns: List[str] = None,
                   date_column: str = None) -> Dict:
    """
    快速验证DataFrame
    
    参数:
        df: 要验证的DataFrame
        required_columns: 必需列
        date_column: 日期列
        
    返回:
        验证结果
    """
    validator = DataValidator()
    return validator.validate_dataframe(df, required_columns, date_column)

def check_lookahead_bias_quick(df: pd.DataFrame,
                              date_column: str,
                              feature_columns: List[str],
                              target_column: str = None) -> Dict:
    """
    快速检查前瞻性偏差
    
    参数:
        df: 数据DataFrame
        date_column: 日期列
        feature_columns: 特征列
        target_column: 目标列
        
    返回:
        前瞻性偏差检查结果
    """
    validator = DataValidator()
    return validator.check_lookahead_bias(df, date_column, feature_columns, target_column)

# 示例使用
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'price': np.random.uniform(100, 200, 100),
        'volume': np.random.randint(1000000, 5000000, 100),
        'return': np.random.normal(0, 0.02, 100)
    }
    df = pd.DataFrame(data)
    
    # 创建验证器
    validator = DataValidator()
    
    # 基本验证
    basic_results = validator.validate_dataframe(
        df, 
        required_columns=['date', 'price', 'volume'],
        date_column='date'
    )
    print("基本验证结果:", basic_results['valid'])
    
    # 前瞻性偏差检查
    lookahead_results = validator.check_lookahead_bias(
        df,
        date_column='date',
        feature_columns=['price', 'volume'],
        target_column='return'
    )
    print("前瞻性偏差检查结果:", lookahead_results['valid'])
    
    # 生成报告
    report = validator.generate_validation_report()
    print(report) 