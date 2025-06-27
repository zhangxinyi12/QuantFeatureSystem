#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征构造全流程脚本
提供完整的数据预处理、特征工程、质量检查、报告生成流程
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.monitoring import SystemMonitor, MemoryManager, PerformanceProfiler
from utils.data_validator import DataValidator
from utils.report_generator import ReportGenerator
from utils.drift_detection import DriftDetector
from utils.date_utils import TradingCalendar, convert_to_trading_date
from processing.feature_engine.price_features import QuantFeatureEngine

class FeaturePipeline:
    """特征构造流水线"""
    
    def __init__(self, config: Dict = None):
        """
        初始化特征流水线
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.monitor = SystemMonitor()
        self.memory_manager = MemoryManager()
        self.validator = DataValidator()
        self.report_generator = ReportGenerator()
        self.drift_detector = DriftDetector()
        self.calendar = TradingCalendar()
        self.profiler = PerformanceProfiler()
        
        # 初始化特征引擎
        self.price_engine = QuantFeatureEngine()
        
        # 流水线状态
        self.pipeline_status = {
            'start_time': None,
            'end_time': None,
            'steps_completed': [],
            'errors': [],
            'warnings': []
        }
        
    def setup_logging(self):
        """设置日志配置"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"feature_pipeline_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def load_data(self, data_path: str, **kwargs) -> pd.DataFrame:
        """
        加载数据
        
        参数:
            data_path: 数据文件路径
            **kwargs: 传递给pd.read_csv的参数
            
        返回:
            加载的数据DataFrame
        """
        self.logger.info(f"开始加载数据: {data_path}")
        
        try:
            # 根据文件扩展名选择加载方法
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path, **kwargs)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path, **kwargs)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                df = pd.read_excel(data_path, **kwargs)
            else:
                raise ValueError(f"不支持的文件格式: {data_path}")
                
            self.logger.info(f"数据加载成功: {df.shape}")
            self.pipeline_status['steps_completed'].append('load_data')
            
            return df
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            self.pipeline_status['errors'].append(f"数据加载失败: {e}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        参数:
            df: 输入数据
            
        返回:
            预处理后的数据
        """
        self.logger.info("开始数据预处理...")
        
        try:
            df_processed = df.copy()
            
            # 1. 处理日期列
            if 'date' in df_processed.columns:
                df_processed['date'] = pd.to_datetime(df_processed['date'])
                df_processed = df_processed.sort_values('date').reset_index(drop=True)
                
            # 2. 处理缺失值
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df_processed[col].isnull().sum() > 0:
                    # 对于价格数据，使用前向填充
                    if col in ['open', 'high', 'low', 'close']:
                        df_processed[col] = df_processed[col].fillna(method='ffill')
                    # 对于成交量数据，使用0填充
                    elif 'volume' in col.lower():
                        df_processed[col] = df_processed[col].fillna(0)
                    # 其他数值列使用中位数填充
                    else:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                        
            # 3. 处理异常值
            for col in numeric_columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    # 使用IQR方法处理异常值
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # 将异常值替换为边界值
                    df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                    
            # 4. 确保价格数据的逻辑一致性
            if all(col in df_processed.columns for col in ['open', 'high', 'low', 'close']):
                # 确保 high >= max(open, close)
                df_processed['high'] = df_processed[['high', 'open', 'close']].max(axis=1)
                # 确保 low <= min(open, close)
                df_processed['low'] = df_processed[['low', 'open', 'close']].min(axis=1)
                
            self.logger.info("数据预处理完成")
            self.pipeline_status['steps_completed'].append('preprocess_data')
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            self.pipeline_status['errors'].append(f"数据预处理失败: {e}")
            raise
            
    def validate_data(self, df: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """
        验证数据质量
        
        参数:
            df: 数据DataFrame
            required_columns: 必需列名列表
            
        返回:
            验证是否通过
        """
        self.logger.info("开始数据验证...")
        
        try:
            if required_columns is None:
                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                
            # 基本验证
            validation_result = self.validator.validate_dataframe(
                df, required_columns=required_columns, date_column='date'
            )
            
            if not validation_result['valid']:
                self.logger.error("数据验证失败")
                for error in validation_result['errors']:
                    self.logger.error(f"错误: {error}")
                self.pipeline_status['errors'].extend(validation_result['errors'])
                return False
                
            # 前瞻性偏差检查
            if 'date' in df.columns:
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                lookahead_result = self.validator.check_lookahead_bias(
                    df, 'date', feature_columns
                )
                
                if not lookahead_result['valid']:
                    self.logger.warning("发现前瞻性偏差问题")
                    self.pipeline_status['warnings'].extend(lookahead_result['lookahead_issues'])
                    
            # 数据一致性检查
            consistency_result = self.validator.check_data_consistency(df)
            if not consistency_result['valid']:
                self.logger.warning("发现数据一致性问题")
                self.pipeline_status['warnings'].extend(consistency_result['consistency_issues'])
                
            self.logger.info("数据验证完成")
            self.pipeline_status['steps_completed'].append('validate_data')
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            self.pipeline_status['errors'].append(f"数据验证失败: {e}")
            return False
            
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成特征
        
        参数:
            df: 预处理后的数据
            
        返回:
            包含特征的数据DataFrame
        """
        self.logger.info("开始生成特征...")
        
        try:
            df_with_features = df.copy()
            
            # 1. 生成价格特征
            self.logger.info("生成价格特征...")
            price_features = self.profiler.profile_function(
                self.price_engine.calculate_all_features, df
            )[0]
            
            if price_features is not None:
                df_with_features = pd.concat([df_with_features, price_features], axis=1)
                self.logger.info(f"价格特征生成完成: {len(price_features.columns)} 个特征")
                
            # 内存优化
            # self.memory_optimizer.optimize_dataframe(df_with_features)  # 注释掉不存在的模块
            
            self.logger.info(f"特征生成完成，总共 {len(df_with_features.columns)} 个特征")
            self.pipeline_status['steps_completed'].append('generate_features')
            
            return df_with_features
            
        except Exception as e:
            self.logger.error(f"特征生成失败: {e}")
            self.pipeline_status['errors'].append(f"特征生成失败: {e}")
            raise
            
    def post_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征后处理
        
        参数:
            df: 包含特征的数据
            
        返回:
            后处理后的数据
        """
        self.logger.info("开始特征后处理...")
        
        try:
            df_processed = df.copy()
            
            # 1. 处理特征中的无穷值和NaN
            feature_columns = [col for col in df_processed.columns 
                             if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
            
            for col in feature_columns:
                if col in df_processed.columns:
                    # 替换无穷值
                    df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                    # 使用中位数填充NaN
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    
            # 2. 特征标准化（可选）
            if self.config.get('standardize_features', False):
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df_processed[feature_columns] = scaler.fit_transform(df_processed[feature_columns])
                
            # 3. 移除低方差特征
            if self.config.get('remove_low_variance', True):
                from sklearn.feature_selection import VarianceThreshold
                selector = VarianceThreshold(threshold=0.01)
                feature_data = df_processed[feature_columns]
                selected_features = selector.fit_transform(feature_data)
                selected_feature_names = [feature_columns[i] for i in range(len(feature_columns)) 
                                        if selector.get_support()[i]]
                
                # 保留原始列和选中的特征
                base_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                base_columns = [col for col in base_columns if col in df_processed.columns]
                df_processed = df_processed[base_columns + selected_feature_names]
                
                self.logger.info(f"移除低方差特征后剩余 {len(selected_feature_names)} 个特征")
                
            self.logger.info("特征后处理完成")
            self.pipeline_status['steps_completed'].append('post_process_features')
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"特征后处理失败: {e}")
            self.pipeline_status['errors'].append(f"特征后处理失败: {e}")
            raise
            
    def generate_reports(self, df: pd.DataFrame, output_dir: str = "output/reports"):
        """
        生成分析报告
        
        参数:
            df: 最终数据
            output_dir: 输出目录
        """
        self.logger.info("开始生成报告...")
        
        try:
            # 生成数据摘要报告
            data_report = self.report_generator.generate_data_summary_report(df)
            self.logger.info("数据摘要报告已生成")
            
            # 生成特征分析报告
            feature_columns = [col for col in df.columns 
                             if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
            
            if feature_columns:
                feature_report = self.report_generator.generate_feature_analysis_report(
                    df, feature_columns
                )
                self.logger.info("特征分析报告已生成")
                
                # 生成可视化报告
                viz_path = self.report_generator.generate_visualization_report(
                    df, feature_columns[:10]  # 只显示前10个特征
                )
                self.logger.info(f"可视化报告已生成: {viz_path}")
                
                # 生成相关性热力图
                corr_path = self.report_generator.generate_correlation_heatmap(
                    df, columns=feature_columns
                )
                self.logger.info(f"相关性热力图已生成: {corr_path}")
                
            # 生成性能报告
            performance_stats = self.profiler.get_all_stats()
            if not performance_stats.empty:
                performance_metrics = {
                    'avg_execution_time': performance_stats['avg_execution_time'].mean(),
                    'total_memory_delta': performance_stats['total_memory_delta'].sum(),
                    'total_calls': performance_stats['call_count'].sum()
                }
                
                performance_report = self.report_generator.generate_performance_report(
                    performance_metrics,
                    model_info={'pipeline_steps': len(self.pipeline_status['steps_completed'])}
                )
                self.logger.info("性能报告已生成")
                
            self.pipeline_status['steps_completed'].append('generate_reports')
            
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            self.pipeline_status['errors'].append(f"报告生成失败: {e}")
            
    def save_results(self, df: pd.DataFrame, output_path: str = None):
        """
        保存结果
        
        参数:
            df: 最终数据
            output_path: 输出路径
        """
        self.logger.info("开始保存结果...")
        
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"output/processed_data/features_{timestamp}.csv"
                
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存为CSV
            df.to_csv(output_path, index=False)
            self.logger.info(f"结果已保存到: {output_path}")
            
            # 同时保存为Parquet格式（更高效）
            parquet_path = output_path.replace('.csv', '.parquet')
            df.to_parquet(parquet_path, index=False)
            self.logger.info(f"结果已保存到: {parquet_path}")
            
            # 保存流水线状态
            status_path = output_path.replace('.csv', '_status.json')
            import json
            with open(status_path, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_status, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"流水线状态已保存到: {status_path}")
            
            self.pipeline_status['steps_completed'].append('save_results')
            
        except Exception as e:
            self.logger.error(f"结果保存失败: {e}")
            self.pipeline_status['errors'].append(f"结果保存失败: {e}")
            
    def run_pipeline(self, data_path: str, output_path: str = None, 
                    config: Dict = None) -> pd.DataFrame:
        """
        运行完整的特征流水线
        
        参数:
            data_path: 输入数据路径
            output_path: 输出路径
            config: 配置参数
            
        返回:
            处理后的数据DataFrame
        """
        self.pipeline_status['start_time'] = datetime.now()
        self.logger.info("开始运行特征流水线...")
        
        try:
            # 更新配置
            if config:
                self.config.update(config)
                
            # 1. 加载数据
            df = self.load_data(data_path)
            
            # 2. 数据预处理
            df = self.preprocess_data(df)
            
            # 3. 数据验证
            if not self.validate_data(df):
                raise ValueError("数据验证失败")
                
            # 4. 生成特征
            df = self.generate_features(df)
            
            # 5. 特征后处理
            df = self.post_process_features(df)
            
            # 6. 生成报告
            self.generate_reports(df)
            
            # 7. 保存结果
            self.save_results(df, output_path)
            
            self.pipeline_status['end_time'] = datetime.now()
            duration = self.pipeline_status['end_time'] - self.pipeline_status['start_time']
            
            self.logger.info(f"特征流水线运行完成，耗时: {duration}")
            self.logger.info(f"完成步骤: {self.pipeline_status['steps_completed']}")
            
            if self.pipeline_status['errors']:
                self.logger.warning(f"发现错误: {self.pipeline_status['errors']}")
                
            if self.pipeline_status['warnings']:
                self.logger.warning(f"发现警告: {self.pipeline_status['warnings']}")
                
            return df
            
        except Exception as e:
            self.pipeline_status['end_time'] = datetime.now()
            self.logger.error(f"特征流水线运行失败: {e}")
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='特征构造流水线')
    parser.add_argument('--input', type=str, required=True, help='输入数据文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径（JSON格式）')
    parser.add_argument('--standardize', action='store_true', help='是否标准化特征')
    parser.add_argument('--remove-low-variance', action='store_true', default=True, 
                       help='是否移除低方差特征')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config:
        import json
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
    config.update({
        'standardize_features': args.standardize,
        'remove_low_variance': args.remove_low_variance
    })
    
    # 创建流水线
    pipeline = FeaturePipeline(config)
    
    try:
        # 运行流水线
        result_df = pipeline.run_pipeline(args.input, args.output, config)
        print(f"流水线运行成功，输出数据形状: {result_df.shape}")
        
    except Exception as e:
        print(f"流水线运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 