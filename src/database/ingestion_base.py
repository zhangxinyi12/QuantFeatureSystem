"""
数据摄入基础模块
提供通用的数据加载和预处理接口
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataIngestionBase(ABC):
    """数据摄入基础类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据摄入器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.data_cache = {}
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        加载数据的抽象方法
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        pass
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据质量的抽象方法
        
        Args:
            df: 待验证的数据
            
        Returns:
            bool: 验证结果
        """
        pass
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        logger.info("开始数据预处理")
        
        # 数据清洗
        df = self.clean_data(df)
        
        # 缺失值处理
        df = self.handle_missing_values(df)
        
        # 异常值处理
        df = self.handle_outliers(df)
        
        # 数据类型转换
        df = self.convert_data_types(df)
        
        # 数据标准化
        df = self.normalize_data(df)
        
        logger.info("数据预处理完成")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        logger.info("开始数据清洗")
        
        # 删除重复行
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.info(f"删除了 {initial_rows - len(df)} 行重复数据")
        
        # 删除全为空的行
        df = df.dropna(how='all')
        
        # 删除全为空的列
        df = df.dropna(axis=1, how='all')
        
        # 处理日期列
        df = self.process_date_columns(df)
        
        # 处理数值列
        df = self.process_numeric_columns(df)
        
        logger.info("数据清洗完成")
        return df
    
    def process_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理日期列
        
        Args:
            df: 数据DataFrame
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 常见的日期列名
        date_columns = ['date', 'Date', 'DATE', 'trading_day', 'TradingDay', 'time', 'Time']
        
        for col in df.columns:
            if col in date_columns or 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"转换日期列: {col}")
                except Exception as e:
                    logger.warning(f"日期列转换失败 {col}: {e}")
        
        return df
    
    def process_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理数值列
        
        Args:
            df: 数据DataFrame
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 常见的价格和成交量列名
        price_columns = ['open', 'high', 'low', 'close', 'price', 'volume', 'amount']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in price_columns):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"转换数值列: {col}")
                except Exception as e:
                    logger.warning(f"数值列转换失败 {col}: {e}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 数据DataFrame
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        logger.info("开始处理缺失值")
        
        # 统计缺失值
        missing_stats = df.isnull().sum()
        if missing_stats.sum() > 0:
            logger.info(f"缺失值统计:\n{missing_stats[missing_stats > 0]}")
        
        # 对于时间序列数据，使用前向填充
        df = df.fillna(method='ffill')
        
        # 对于剩余的缺失值，使用后向填充
        df = df.fillna(method='bfill')
        
        # 对于数值列，使用均值填充
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
        
        logger.info("缺失值处理完成")
        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            df: 数据DataFrame
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        logger.info("开始处理异常值")
        
        # 只处理数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # 计算IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # 定义异常值边界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 统计异常值
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                logger.info(f"列 {col} 发现 {outliers} 个异常值")
                
                # 将异常值替换为边界值
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info("异常值处理完成")
        return df
    
    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据类型
        
        Args:
            df: 数据DataFrame
            
        Returns:
            pd.DataFrame: 转换后的数据
        """
        logger.info("开始转换数据类型")
        
        # 优化数值类型
        for col in df.select_dtypes(include=['float64']).columns:
            if df[col].notna().all():
                # 检查是否可以转换为整数
                if (df[col] % 1 == 0).all():
                    df[col] = df[col].astype('int64')
                    logger.info(f"列 {col} 转换为整数类型")
        
        # 优化字符串类型
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # 低基数列
                df[col] = df[col].astype('category')
                logger.info(f"列 {col} 转换为分类类型")
        
        logger.info("数据类型转换完成")
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据标准化
        
        Args:
            df: 数据DataFrame
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        logger.info("开始数据标准化")
        
        # 只对数值列进行标准化
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # 跳过已经是标准化的列
            if 'normalized' in col.lower() or 'scaled' in col.lower():
                continue
            
            # Z-score标准化
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                df[f'{col}_normalized'] = (df[col] - mean_val) / std_val
                logger.info(f"列 {col} 标准化完成")
        
        logger.info("数据标准化完成")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取数据摘要
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict: 数据摘要信息
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # 数值列摘要
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            summary['numeric_summary'] = df[numeric_columns].describe().to_dict()
        
        # 分类列摘要
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                summary['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
        
        return summary
    
    def save_data(self, df: pd.DataFrame, filepath: Union[str, Path], 
                  format: str = 'csv') -> None:
        """
        保存数据
        
        Args:
            df: 要保存的数据
            filepath: 文件路径
            format: 文件格式 ('csv', 'parquet', 'feather')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=False)
        elif format.lower() == 'feather':
            df.to_feather(filepath, index=False)
        else:
            raise ValueError(f"不支持的文件格式: {format}")
        
        logger.info(f"数据已保存到: {filepath}")
    
    def load_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[pd.DataFrame]: 缓存的数据
        """
        return self.data_cache.get(key)
    
    def cache_data(self, key: str, df: pd.DataFrame) -> None:
        """
        缓存数据
        
        Args:
            key: 缓存键
            df: 要缓存的数据
        """
        self.data_cache[key] = df.copy()
        logger.info(f"数据已缓存: {key}")
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.data_cache.clear()
        logger.info("缓存已清空")
