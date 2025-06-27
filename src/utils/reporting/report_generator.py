#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成工具模块
提供数据报告、特征报告、性能报告等生成功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import os
import json
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = "output/reports", logger: Optional[logging.Logger] = None):
        """
        初始化报告生成器
        
        参数:
            output_dir: 输出目录
            logger: 日志记录器
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
    def generate_data_summary_report(self, df: pd.DataFrame, 
                                   title: str = "数据摘要报告",
                                   save_path: Optional[str] = None) -> str:
        """
        生成数据摘要报告
        
        参数:
            df: 数据DataFrame
            title: 报告标题
            save_path: 保存路径
            
        返回:
            报告内容
        """
        report = f"{title}\n"
        report += "=" * 50 + "\n\n"
        
        # 基本信息
        report += "1. 基本信息\n"
        report += "-" * 20 + "\n"
        report += f"数据形状: {df.shape}\n"
        report += f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
        report += f"数据类型数量: {len(df.dtypes.value_counts())}\n\n"
        
        # 数据类型信息
        report += "2. 数据类型分布\n"
        report += "-" * 20 + "\n"
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            report += f"{dtype}: {count} 列\n"
        report += "\n"
        
        # 缺失值信息
        report += "3. 缺失值统计\n"
        report += "-" * 20 + "\n"
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df)) * 100
        
        missing_df = pd.DataFrame({
            '缺失数量': missing_counts,
            '缺失比例(%)': missing_percent
        }).sort_values('缺失数量', ascending=False)
        
        missing_df = missing_df[missing_df['缺失数量'] > 0]
        
        if not missing_df.empty:
            for col in missing_df.index:
                report += f"{col}: {missing_df.loc[col, '缺失数量']} ({missing_df.loc[col, '缺失比例(%)']:.2f}%)\n"
        else:
            report += "无缺失值\n"
        report += "\n"
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report += "4. 数值列统计\n"
            report += "-" * 20 + "\n"
            numeric_stats = df[numeric_cols].describe()
            report += numeric_stats.to_string()
            report += "\n\n"
            
        # 分类列统计
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            report += "5. 分类列统计\n"
            report += "-" * 20 + "\n"
            for col in categorical_cols:
                unique_count = df[col].nunique()
                report += f"{col}: {unique_count} 个唯一值\n"
                if unique_count <= 10:  # 只显示少量类别的分布
                    value_counts = df[col].value_counts()
                    for value, count in value_counts.items():
                        report += f"  {value}: {count} ({count/len(df)*100:.2f}%)\n"
                report += "\n"
                
        # 保存报告
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"data_summary_report_{timestamp}.txt"
        else:
            save_path = Path(save_path)
            
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"数据摘要报告已保存到: {save_path}")
        return report
        
    def generate_feature_analysis_report(self, df: pd.DataFrame,
                                       feature_columns: List[str],
                                       target_column: Optional[str] = None,
                                       title: str = "特征分析报告",
                                       save_path: Optional[str] = None) -> str:
        """
        生成特征分析报告
        
        参数:
            df: 数据DataFrame
            feature_columns: 特征列列表
            target_column: 目标列（可选）
            title: 报告标题
            save_path: 保存路径
            
        返回:
            报告内容
        """
        report = f"{title}\n"
        report += "=" * 50 + "\n\n"
        
        # 特征基本信息
        report += "1. 特征基本信息\n"
        report += "-" * 20 + "\n"
        report += f"特征数量: {len(feature_columns)}\n"
        report += f"样本数量: {len(df)}\n\n"
        
        # 特征统计信息
        report += "2. 特征统计信息\n"
        report += "-" * 20 + "\n"
        
        for feature in feature_columns:
            if feature not in df.columns:
                report += f"{feature}: 列不存在\n"
                continue
                
            series = df[feature]
            report += f"\n{feature}:\n"
            report += f"  数据类型: {series.dtype}\n"
            report += f"  缺失值: {series.isnull().sum()} ({series.isnull().sum()/len(series)*100:.2f}%)\n"
            
            if pd.api.types.is_numeric_dtype(series):
                report += f"  均值: {series.mean():.4f}\n"
                report += f"  标准差: {series.std():.4f}\n"
                report += f"  最小值: {series.min():.4f}\n"
                report += f"  最大值: {series.max():.4f}\n"
                report += f"  中位数: {series.median():.4f}\n"
                
                # 检查异常值
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                outliers = series[(series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)]
                report += f"  异常值数量: {len(outliers)} ({len(outliers)/len(series)*100:.2f}%)\n"
                
            elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
                unique_count = series.nunique()
                report += f"  唯一值数量: {unique_count}\n"
                if unique_count <= 10:
                    value_counts = series.value_counts()
                    report += f"  值分布:\n"
                    for value, count in value_counts.items():
                        report += f"    {value}: {count} ({count/len(series)*100:.2f}%)\n"
                        
        # 特征与目标关系分析
        if target_column and target_column in df.columns:
            report += "\n3. 特征与目标关系分析\n"
            report += "-" * 20 + "\n"
            
            target_series = df[target_column]
            
            for feature in feature_columns:
                if feature not in df.columns:
                    continue
                    
                feature_series = df[feature]
                
                # 计算相关性
                if pd.api.types.is_numeric_dtype(feature_series) and pd.api.types.is_numeric_dtype(target_series):
                    correlation = feature_series.corr(target_series)
                    report += f"\n{feature} 与 {target_column} 的相关系数: {correlation:.4f}\n"
                    
                    # 相关性强度判断
                    if abs(correlation) >= 0.7:
                        strength = "强"
                    elif abs(correlation) >= 0.3:
                        strength = "中等"
                    else:
                        strength = "弱"
                    report += f"  相关性强度: {strength}\n"
                    
        # 保存报告
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"feature_analysis_report_{timestamp}.txt"
        else:
            save_path = Path(save_path)
            
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"特征分析报告已保存到: {save_path}")
        return report
        
    def generate_performance_report(self, 
                                  performance_metrics: Dict,
                                  model_info: Dict = None,
                                  title: str = "模型性能报告",
                                  save_path: Optional[str] = None) -> str:
        """
        生成模型性能报告
        
        参数:
            performance_metrics: 性能指标字典
            model_info: 模型信息字典
            title: 报告标题
            save_path: 保存路径
            
        返回:
            报告内容
        """
        report = f"{title}\n"
        report += "=" * 50 + "\n\n"
        
        # 模型信息
        if model_info:
            report += "1. 模型信息\n"
            report += "-" * 20 + "\n"
            for key, value in model_info.items():
                report += f"{key}: {value}\n"
            report += "\n"
            
        # 性能指标
        report += "2. 性能指标\n"
        report += "-" * 20 + "\n"
        
        for metric_name, metric_value in performance_metrics.items():
            if isinstance(metric_value, float):
                report += f"{metric_name}: {metric_value:.4f}\n"
            else:
                report += f"{metric_name}: {metric_value}\n"
        report += "\n"
        
        # 性能评估
        report += "3. 性能评估\n"
        report += "-" * 20 + "\n"
        
        # 根据常见指标进行评估
        if 'accuracy' in performance_metrics:
            accuracy = performance_metrics['accuracy']
            if accuracy >= 0.9:
                evaluation = "优秀"
            elif accuracy >= 0.8:
                evaluation = "良好"
            elif accuracy >= 0.7:
                evaluation = "一般"
            else:
                evaluation = "需要改进"
            report += f"准确率评估: {evaluation} ({accuracy:.2%})\n"
            
        if 'precision' in performance_metrics and 'recall' in performance_metrics:
            precision = performance_metrics['precision']
            recall = performance_metrics['recall']
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            report += f"F1分数: {f1_score:.4f}\n"
            
        if 'rmse' in performance_metrics:
            rmse = performance_metrics['rmse']
            report += f"RMSE: {rmse:.4f}\n"
            
        # 保存报告
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"performance_report_{timestamp}.txt"
        else:
            save_path = Path(save_path)
            
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"性能报告已保存到: {save_path}")
        return report
        
    def generate_visualization_report(self, df: pd.DataFrame,
                                    feature_columns: List[str],
                                    target_column: Optional[str] = None,
                                    save_path: Optional[str] = None) -> str:
        """
        生成可视化报告
        
        参数:
            df: 数据DataFrame
            feature_columns: 特征列列表
            target_column: 目标列（可选）
            save_path: 保存路径
            
        返回:
            图表保存路径
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"visualization_report_{timestamp}.png"
        else:
            save_path = Path(save_path)
            
        # 计算子图数量
        n_features = len(feature_columns)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # 创建图表
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
            
        # 绘制每个特征的分布
        for i, feature in enumerate(feature_columns):
            if feature not in df.columns:
                continue
                
            ax = axes[i]
            series = df[feature].dropna()
            
            if pd.api.types.is_numeric_dtype(series):
                # 数值特征：直方图
                ax.hist(series, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{feature} 分布')
                ax.set_xlabel(feature)
                ax.set_ylabel('频数')
                
                # 添加统计信息
                mean_val = series.mean()
                std_val = series.std()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.2f}')
                ax.legend()
                
            else:
                # 分类特征：条形图
                value_counts = series.value_counts()
                if len(value_counts) <= 20:  # 只显示前20个类别
                    value_counts.head(20).plot(kind='bar', ax=ax)
                    ax.set_title(f'{feature} 分布')
                    ax.set_xlabel(feature)
                    ax.set_ylabel('频数')
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                    
        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"可视化报告已保存到: {save_path}")
        return str(save_path)
        
    def generate_correlation_heatmap(self, df: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   save_path: Optional[str] = None) -> str:
        """
        生成相关性热力图
        
        参数:
            df: 数据DataFrame
            columns: 要分析的列（默认所有数值列）
            save_path: 保存路径
            
        返回:
            图表保存路径
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        # 计算相关性矩阵
        corr_matrix = df[columns].corr()
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('特征相关性热力图')
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"correlation_heatmap_{timestamp}.png"
        else:
            save_path = Path(save_path)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"相关性热力图已保存到: {save_path}")
        return str(save_path)
        
    def generate_comprehensive_report(self, df: pd.DataFrame,
                                    feature_columns: List[str],
                                    target_column: Optional[str] = None,
                                    performance_metrics: Optional[Dict] = None,
                                    model_info: Optional[Dict] = None,
                                    title: str = "综合分析报告") -> Dict:
        """
        生成综合分析报告
        
        参数:
            df: 数据DataFrame
            feature_columns: 特征列列表
            target_column: 目标列（可选）
            performance_metrics: 性能指标（可选）
            model_info: 模型信息（可选）
            title: 报告标题
            
        返回:
            报告文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}
        
        # 生成数据摘要报告
        data_summary_path = self.output_dir / f"data_summary_{timestamp}.txt"
        self.generate_data_summary_report(df, title="数据摘要报告", save_path=data_summary_path)
        report_files['data_summary'] = str(data_summary_path)
        
        # 生成特征分析报告
        feature_analysis_path = self.output_dir / f"feature_analysis_{timestamp}.txt"
        self.generate_feature_analysis_report(df, feature_columns, target_column, 
                                            title="特征分析报告", save_path=feature_analysis_path)
        report_files['feature_analysis'] = str(feature_analysis_path)
        
        # 生成可视化报告
        visualization_path = self.output_dir / f"visualization_{timestamp}.png"
        self.generate_visualization_report(df, feature_columns, target_column, save_path=visualization_path)
        report_files['visualization'] = str(visualization_path)
        
        # 生成相关性热力图
        correlation_path = self.output_dir / f"correlation_{timestamp}.png"
        self.generate_correlation_heatmap(df, columns=feature_columns, save_path=correlation_path)
        report_files['correlation'] = str(correlation_path)
        
        # 生成性能报告（如果有）
        if performance_metrics:
            performance_path = self.output_dir / f"performance_{timestamp}.txt"
            self.generate_performance_report(performance_metrics, model_info, 
                                           title="模型性能报告", save_path=performance_path)
            report_files['performance'] = str(performance_path)
            
        # 生成报告索引
        index_path = self.output_dir / f"report_index_{timestamp}.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(report_files, f, ensure_ascii=False, indent=2)
        report_files['index'] = str(index_path)
        
        self.logger.info(f"综合分析报告已生成，索引文件: {index_path}")
        return report_files

# 便捷函数
def quick_data_report(df: pd.DataFrame, output_dir: str = "output/reports") -> str:
    """快速生成数据报告"""
    generator = ReportGenerator(output_dir)
    return generator.generate_data_summary_report(df)

def quick_feature_report(df: pd.DataFrame, feature_columns: List[str], 
                        target_column: str = None, output_dir: str = "output/reports") -> str:
    """快速生成特征报告"""
    generator = ReportGenerator(output_dir)
    return generator.generate_feature_analysis_report(df, feature_columns, target_column)

# 示例使用
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'price': np.random.uniform(100, 200, 100),
        'volume': np.random.randint(1000000, 5000000, 100),
        'return': np.random.normal(0, 0.02, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    df = pd.DataFrame(data)
    
    # 创建报告生成器
    generator = ReportGenerator()
    
    # 生成数据摘要报告
    data_report = generator.generate_data_summary_report(df)
    print("数据摘要报告已生成")
    
    # 生成特征分析报告
    feature_report = generator.generate_feature_analysis_report(
        df, ['price', 'volume'], 'return'
    )
    print("特征分析报告已生成")
    
    # 生成可视化报告
    viz_path = generator.generate_visualization_report(df, ['price', 'volume', 'return'])
    print(f"可视化报告已生成: {viz_path}")
    
    # 生成相关性热力图
    corr_path = generator.generate_correlation_heatmap(df)
    print(f"相关性热力图已生成: {corr_path}") 