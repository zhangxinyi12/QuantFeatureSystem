#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化特征系统主程序入口
统一的数据处理、特征工程、监控和分析系统
"""

import sys
import os
import logging.config
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入配置
try:
    from config.settings import DATA_CONFIG, OUTPUT_CONFIG, LOGGING_CONFIG
except ImportError:
    # 如果配置文件不存在，使用默认配置
    DATA_CONFIG = {
        'default_start_date': '2024-01-01',
        'default_end_date': '2024-12-31',
        'market_codes': [1, 2],
        'output_format': 'parquet',
        'chunk_size': 10000
    }
    OUTPUT_CONFIG = {
        'processed_data_dir': Path('output/processed_data'),
        'reports_dir': Path('output/reports'),
        'logs_dir': Path('logs')
    }

# 导入核心模块
try:
    from database.connector import JuyuanDB
    from database.queries import StockQueries
    from processing.timeseries import TimeSeriesProcessor
    from processing.feature_engine.price_features import QuantFeatureEngine
    from utils.memory_utils import MemoryManager, DataFrameChunker, MemoryMonitor
    from utils.monitoring import SystemMonitor
    from utils.data_validator import DataValidator
    from utils.report_generator import ReportGenerator
    from utils.drift_detection import DriftDetector
    from utils.date_utils import TradingCalendar
except ImportError as e:
    print(f"警告: 某些模块导入失败: {e}")
    print("将使用基础功能模式")

from .preprocess import data_validator, date_utils, drift_detection, market_features
from .monitoring import memory_utils, monitoring
from .reporting import report_generator

class QuantFeatureSystem:
    """量化特征系统主类"""
    
    def __init__(self):
        """初始化系统"""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 创建输出目录
        self.create_output_directories()
        
        # 初始化组件
        self.init_components()
        
    def setup_logging(self):
        """设置日志配置"""
        log_dir = OUTPUT_CONFIG.get('logs_dir', Path('logs'))
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"quant_feature_system_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def create_output_directories(self):
        """创建输出目录"""
        directories = [
            OUTPUT_CONFIG.get('processed_data_dir', Path('output/processed_data')),
            OUTPUT_CONFIG.get('reports_dir', Path('output/reports')),
            OUTPUT_CONFIG.get('logs_dir', Path('logs'))
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def init_components(self):
        """初始化系统组件"""
        try:
            self.memory_manager = MemoryManager()
            self.memory_monitor = MemoryMonitor()
            self.system_monitor = SystemMonitor()
            self.validator = DataValidator()
            self.report_generator = ReportGenerator()
            self.drift_detector = DriftDetector()
            self.calendar = TradingCalendar()
            
            # 初始化特征引擎
            self.volume_engine = QuantFeatureEngine()
            
            self.logger.info("所有组件初始化成功")
        except Exception as e:
            self.logger.warning(f"部分组件初始化失败: {e}")
            
    def process_stock_data(self, args):
        """处理股票数据（从数据库查询到特征生成）"""
        self.logger.info("开始执行股票数据处理")
        self.logger.info(f"参数: {vars(args)}")
        
        try:
            # 连接数据库
            self.logger.info("连接数据库...")
            with JuyuanDB() as db:
                if not db.test_connection():
                    self.logger.error("数据库连接测试失败")
                    return None
                
                self.logger.info("数据库连接成功")
                
                # 测试模式：只查询最近一周的数据
                if args.test_mode:
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=7)
                    args.start_date = start_date.strftime('%Y-%m-%d')
                    args.end_date = end_date.strftime('%Y-%m-%d')
                    self.logger.info(f"测试模式：查询 {args.start_date} 到 {args.end_date} 的数据")
                
                # 构建查询SQL
                sql = StockQueries.get_stock_quote_history(
                    start_date=args.start_date,
                    end_date=args.end_date,
                    market_codes=args.market_codes,
                    include_suspended=False
                )
                
                self.logger.info("开始查询数据...")
                df = db.read_sql(sql)
                
                if df.empty:
                    self.logger.warning("查询结果为空")
                    return None
                
                self.logger.info(f"查询完成，共获取 {len(df)} 行数据")
                
                # 数据处理和特征生成
                df = self.process_and_generate_features(df, args)
                
                # 保存数据
                self.save_processed_data(df, args)
                
                # 生成报告
                self.generate_processing_reports(df, args)
                
                return df
                
        except Exception as e:
            self.logger.error(f"数据处理失败: {e}", exc_info=True)
            return None
            
    def process_and_generate_features(self, df, args):
        """处理和生成特征"""
        # 内存优化
        self.logger.info("开始内存优化...")
        df = self.memory_manager.optimize_dataframe_memory(df)
        
        # 时序数据处理
        self.logger.info("开始时序数据处理...")
        processor = TimeSeriesProcessor()
        
        # 价格复权
        if args.adj_type != 'none':
            df = processor.adjust_prices(df, adj_type=args.adj_type)
        
        # 计算技术指标
        if args.include_technical:
            self.logger.info("计算技术指标...")
            df = processor.calculate_technical_indicators(df)
            df = processor.calculate_volume_indicators(df)
            df = processor.calculate_support_resistance(df)
        
        # 计算量化特征
        if args.include_features:
            self.logger.info("计算量化特征...")
            df = self.generate_all_features(df, args.feature_types)
            self.logger.info(f"特征计算完成，共生成 {len(df.columns)} 个特征")
        
        # 过滤数据
        df = processor.filter_trading_days(df)
        
        return df
        
    def generate_all_features(self, df, feature_types):
        """生成所有特征"""
        try:
            # 标准化列名
            df_processed = df.copy()
            
            # 重命名列以匹配特征引擎期望的格式
            column_mapping = {
                'TradingDay': 'date',
                'OpenPrice': 'open',
                'HighPrice': 'high', 
                'LowPrice': 'low',
                'ClosePrice': 'close',
                'TurnoverVolume': 'volume',
                'TurnoverValue': 'value'
            }
            
            df_processed = df_processed.rename(columns=column_mapping)
            
            # 生成不同类型的特征
            if 'volume' in feature_types or 'price' in feature_types:
                # 使用统一的特征引擎处理所有特征类型
                df_processed = self.volume_engine.calculate_all_features(df_processed, feature_types)
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"特征生成失败: {e}")
            return df
            
    def save_processed_data(self, df, args):
        """保存处理后的数据"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if args.output_format == 'feather':
            output_file = OUTPUT_CONFIG['processed_data_dir'] / f'stock_data_{timestamp}.feather'
            df.to_feather(output_file)
        elif args.output_format == 'parquet':
            output_file = OUTPUT_CONFIG['processed_data_dir'] / f'stock_data_{timestamp}.parquet'
            df.to_parquet(output_file, compression='snappy')
        else:  # csv
            output_file = OUTPUT_CONFIG['processed_data_dir'] / f'stock_data_{timestamp}.csv'
            df.to_csv(output_file, index=False)
        
        self.logger.info(f"数据已保存到: {output_file}")
        
    def generate_processing_reports(self, df, args):
        """生成处理报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成数据报告
        data_report = self.generate_data_report(df, args)
        report_file = OUTPUT_CONFIG['reports_dir'] / f'data_report_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(data_report)
        
        self.logger.info(f"数据报告已保存到: {report_file}")
        
        # 生成特征报告
        if args.include_features:
            feature_report = self.generate_feature_report(df, args)
            feature_report_file = OUTPUT_CONFIG['reports_dir'] / f'feature_report_{timestamp}.txt'
            
            with open(feature_report_file, 'w', encoding='utf-8') as f:
                f.write(feature_report)
            
            self.logger.info(f"特征报告已保存到: {feature_report_file}")
            
        # 内存监控报告
        memory_report = self.memory_monitor.generate_memory_report()
        self.logger.info(memory_report)
        
    def start_monitoring(self, duration_minutes: int = 60):
        """启动系统监控"""
        self.logger.info("启动系统监控...")
        self.system_monitor.start_monitoring()
        
        try:
            import time
            time.sleep(duration_minutes * 60)
        except KeyboardInterrupt:
            self.logger.info("收到中断信号，停止监控")
        finally:
            self.system_monitor.stop_monitoring()
            
        # 生成监控报告
        summary = self.system_monitor.get_performance_summary()
        self.logger.info(f"监控摘要: {summary}")
        
    def validate_data(self, data_path: str, required_columns: list = None):
        """验证数据质量"""
        self.logger.info(f"开始验证数据: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            
            # 基本验证
            validation_result = self.validator.validate_dataframe(
                df, required_columns=required_columns
            )
            
            if validation_result['valid']:
                self.logger.info("数据验证通过")
            else:
                self.logger.warning("数据验证失败")
                for error in validation_result['errors']:
                    self.logger.error(f"错误: {error}")
                    
            # 前瞻性偏差检查
            if 'date' in df.columns:
                lookahead_result = self.validator.check_lookahead_bias(
                    df, 'date', df.select_dtypes(include=['number']).columns.tolist()
                )
                
                if lookahead_result['valid']:
                    self.logger.info("前瞻性偏差检查通过")
                else:
                    self.logger.warning("发现前瞻性偏差问题")
                    
            return validation_result
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return None
            
    def generate_features_from_file(self, data_path: str, output_path: str = None, feature_types=None):
        """从文件生成特征"""
        self.logger.info(f"开始生成特征: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            
            # 确保有必要的列
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"缺少必要列: {missing_columns}")
                return None
                
            # 生成特征
            df_with_features = self.volume_engine.calculate_all_features(df, feature_types or ['volume', 'price'])
            
            # 保存结果
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = OUTPUT_CONFIG['processed_data_dir'] / f'features_{timestamp}.csv'
                
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            df_with_features.to_csv(output_path, index=False)
            self.logger.info(f"特征已保存到: {output_path}")
            
            return df_with_features
            
        except Exception as e:
            self.logger.error(f"特征生成失败: {e}")
            return None
            
    def generate_report(self, data_path: str, output_dir: str = None):
        """生成分析报告"""
        self.logger.info(f"开始生成报告: {data_path}")
        
        if output_dir is None:
            output_dir = OUTPUT_CONFIG['reports_dir']
        
        try:
            df = pd.read_csv(data_path)
            
            # 生成数据摘要报告
            data_report = self.report_generator.generate_data_summary_report(df)
            self.logger.info("数据摘要报告已生成")
            
            # 生成特征分析报告
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            feature_report = self.report_generator.generate_feature_analysis_report(
                df, numeric_columns
            )
            self.logger.info("特征分析报告已生成")
            
            # 生成可视化报告
            viz_path = self.report_generator.generate_visualization_report(
                df, numeric_columns[:5]  # 只显示前5个特征
            )
            self.logger.info(f"可视化报告已生成: {viz_path}")
            
            # 生成相关性热力图
            corr_path = self.report_generator.generate_correlation_heatmap(df)
            self.logger.info(f"相关性热力图已生成: {corr_path}")
            
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            
    def detect_drift(self, reference_data_path: str, current_data_path: str):
        """检测数据漂移"""
        self.logger.info("开始检测数据漂移...")
        
        try:
            reference_df = pd.read_csv(reference_data_path)
            current_df = pd.read_csv(current_data_path)
            
            # 设置参考数据
            self.drift_detector.set_reference_data(reference_df)
            
            # 检测分布漂移
            drift_result = self.drift_detector.detect_distribution_drift(current_df)
            
            if drift_result['drift_detected']:
                self.logger.warning("检测到分布漂移")
                self.logger.warning(f"漂移特征: {drift_result['drift_features']}")
            else:
                self.logger.info("未检测到分布漂移")
                
            # 检测协变量漂移
            covariate_result = self.drift_detector.detect_covariate_drift(current_df)
            
            if covariate_result['drift_detected']:
                self.logger.warning("检测到协变量漂移")
                self.logger.warning(f"漂移分数: {covariate_result['drift_score']}")
            else:
                self.logger.info("未检测到协变量漂移")
                
            return drift_result, covariate_result
            
        except Exception as e:
            self.logger.error(f"漂移检测失败: {e}")
            return None, None
            
    def optimize_memory(self):
        """优化内存使用"""
        self.logger.info("开始内存优化...")
        
        before_usage = self.memory_manager.get_memory_usage()
        self.logger.info(f"优化前内存使用: {before_usage['process_memory_mb']:.2f} MB")
        
        optimization_result = self.memory_manager.optimize_memory()
        
        after_usage = optimization_result['after_usage']
        self.logger.info(f"优化后内存使用: {after_usage['process_memory_mb']:.2f} MB")
        self.logger.info(f"释放内存: {optimization_result['freed_mb']:.2f} MB")
        
        return optimization_result
        
    def show_system_info(self):
        """显示系统信息"""
        self.logger.info("系统信息:")
        
        # 系统基本信息
        import psutil
        system_info = {
            'CPU核心数': psutil.cpu_count(),
            '内存总量': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            '磁盘总量': f"{psutil.disk_usage('/').total / (1024**3):.2f} GB",
            'Python版本': sys.version,
            '项目路径': str(project_root)
        }
        
        for key, value in system_info.items():
            self.logger.info(f"  {key}: {value}")
            
        # 内存使用情况
        memory_usage = self.memory_manager.get_memory_usage()
        self.logger.info(f"当前内存使用: {memory_usage['process_memory_mb']:.2f} MB")
        
        # 内存压力评估
        pressure = self.memory_manager.check_memory_pressure()
        self.logger.info(f"内存压力等级: {pressure['pressure_level']}")
        self.logger.info(f"建议: {pressure['recommendation']}")
        
    def generate_data_report(self, df, args):
        """生成数据报告"""
        report = f"""
股票数据处理报告
================

处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
查询参数:
- 开始日期: {args.start_date}
- 结束日期: {args.end_date}
- 市场代码: {args.market_codes}
- 复权类型: {args.adj_type}
- 技术指标: {'是' if args.include_technical else '否'}
- 量化特征: {'是' if args.include_features else '否'}
- 特征类型: {args.feature_types if args.include_features else '无'}

数据统计:
- 总记录数: {len(df):,}
- 股票数量: {df['SecuCode'].nunique():,}
- 交易日期范围: {df['TradingDay'].min()} 到 {df['TradingDay'].max()}
- 数据列数: {len(df.columns)}

数据列信息:
"""
        
        for col in df.columns:
            report += f"- {col}: {df[col].dtype}\n"
        
        if 'ClosePrice' in df.columns:
            report += f"""
价格统计:
- 收盘价范围: {df['ClosePrice'].min():.2f} - {df['ClosePrice'].max():.2f}
- 平均收盘价: {df['ClosePrice'].mean():.2f}
- 收盘价标准差: {df['ClosePrice'].std():.2f}
"""
        
        if 'TurnoverVolume' in df.columns:
            report += f"""
成交量统计:
- 成交量范围: {df['TurnoverVolume'].min():,.0f} - {df['TurnoverVolume'].max():,.0f}
- 平均成交量: {df['TurnoverVolume'].mean():,.0f}
- 总成交量: {df['TurnoverVolume'].sum():,.0f}
"""
        
        return report

    def generate_feature_report(self, df, args):
        """生成特征报告"""
        # 获取新增的特征列
        original_columns = {
            'SecuCode', 'SecuMarket', 'Ifsuspend', 'TradingDay', 
            'PrevClosePrice', 'OpenPrice', 'HighPrice', 'LowPrice', 
            'ClosePrice', 'TurnoverVolume', 'TurnoverValue', 
            'PriceCeiling', 'PriceFloor'
        }
        
        feature_columns = [col for col in df.columns if col not in original_columns]
        
        report = f"""
量化特征报告
============

特征计算时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
特征类型: {args.feature_types}
总特征数: {len(feature_columns)}

特征分类统计:
"""
        
        # 按特征类型分类统计
        feature_categories = {
            '价格特征': [col for col in feature_columns if any(x in col for x in ['Return', 'Momentum', 'Vol', 'SMA', 'EMA', 'BB', 'Support', 'Resistance', 'Fib'])],
            '成交量特征': [col for col in feature_columns if any(x in col for x in ['Volume', 'VWAP', 'MFI'])],
            '技术指标': [col for col in feature_columns if any(x in col for x in ['RSI', 'MACD', 'Stoch', 'Williams', 'Hammer', 'Doji', 'Engulfing'])],
            '季节性特征': [col for col in feature_columns if any(x in col for x in ['DayOfWeek', 'Month', 'Quarter', 'Seasonality'])]
        }
        
        for category, features in feature_categories.items():
            report += f"- {category}: {len(features)} 个\n"
        
        report += f"""
特征列表:
"""
        
        for category, features in feature_categories.items():
            if features:
                report += f"\n{category}:\n"
                for feature in sorted(features):
                    report += f"  - {feature}\n"
        
        # 特征质量统计
        report += f"""
特征质量统计:
- 缺失值比例 > 50% 的特征: {len([col for col in feature_columns if df[col].isna().sum() / len(df) > 0.5])} 个
- 零方差特征: {len([col for col in feature_columns if df[col].std() == 0])} 个
- 常数特征: {len([col for col in feature_columns if df[col].nunique() == 1])} 个

特征相关性分析:
- 高度相关特征对 (>0.95): 需要进一步分析
- 建议进行特征选择以降低维度
"""
        
        return report

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='量化特征系统')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 数据处理命令
    process_parser = subparsers.add_parser('process', help='处理股票数据')
    process_parser.add_argument('--start-date', type=str, default=DATA_CONFIG['default_start_date'], help='开始日期 (YYYY-MM-DD)')
    process_parser.add_argument('--end-date', type=str, default=DATA_CONFIG['default_end_date'], help='结束日期 (YYYY-MM-DD)')
    process_parser.add_argument('--market-codes', type=int, nargs='+', default=DATA_CONFIG['market_codes'], help='市场代码列表')
    process_parser.add_argument('--output-format', type=str, default=DATA_CONFIG['output_format'], choices=['feather', 'parquet', 'csv'], help='输出格式')
    process_parser.add_argument('--chunk-size', type=int, default=DATA_CONFIG['chunk_size'], help='分块大小')
    process_parser.add_argument('--adj-type', type=str, default='forward', choices=['forward', 'backward', 'none'], help='复权类型')
    process_parser.add_argument('--include-technical', action='store_true', help='是否计算技术指标')
    process_parser.add_argument('--include-features', action='store_true', help='是否计算量化特征')
    process_parser.add_argument('--feature-types', type=str, nargs='+', default=['price', 'volume', 'technical'], choices=['price', 'volume', 'technical', 'cross_asset'], help='要计算的特征类型')
    process_parser.add_argument('--test-mode', action='store_true', help='测试模式（只处理少量数据）')
    
    # 监控命令
    monitor_parser = subparsers.add_parser('monitor', help='启动系统监控')
    monitor_parser.add_argument('--duration', type=int, default=60, help='监控持续时间（分钟）')
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='验证数据质量')
    validate_parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    validate_parser.add_argument('--columns', nargs='+', help='必需列名')
    
    # 特征生成命令
    features_parser = subparsers.add_parser('features', help='从文件生成特征')
    features_parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    features_parser.add_argument('--output', type=str, help='输出路径')
    features_parser.add_argument('--feature-types', type=str, nargs='+', default=['volume', 'price'], choices=['volume', 'price', 'technical', 'cross_asset'], help='特征类型')
    
    # 报告生成命令
    report_parser = subparsers.add_parser('report', help='生成分析报告')
    report_parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    report_parser.add_argument('--output', type=str, help='输出目录')
    
    # 漂移检测命令
    drift_parser = subparsers.add_parser('drift', help='检测数据漂移')
    drift_parser.add_argument('--reference', type=str, required=True, help='参考数据路径')
    drift_parser.add_argument('--current', type=str, required=True, help='当前数据路径')
    
    # 内存优化命令
    optimize_parser = subparsers.add_parser('optimize', help='优化内存使用')
    
    # 系统信息命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    if not args.command:
        print("请指定要执行的命令。使用 --help 查看帮助信息。")
        return 1
    
    # 创建系统实例
    system = QuantFeatureSystem()
    
    try:
        if args.command == 'process':
            system.process_stock_data(args)
            
        elif args.command == 'monitor':
            system.start_monitoring(args.duration)
            
        elif args.command == 'validate':
            system.validate_data(args.data, args.columns)
            
        elif args.command == 'features':
            system.generate_features_from_file(args.data, args.output, args.feature_types)
            
        elif args.command == 'report':
            system.generate_report(args.data, args.output)
            
        elif args.command == 'drift':
            system.detect_drift(args.reference, args.current)
            
        elif args.command == 'optimize':
            system.optimize_memory()
            
        elif args.command == 'info':
            system.show_system_info()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        return 1
    except Exception as e:
        print(f"程序执行出错: {e}")
        system.logger.error(f"程序执行出错: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 