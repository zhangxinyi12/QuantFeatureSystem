"""
内存优化工具模块
用于低配服务器的内存管理和优化
"""

import psutil
import gc
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import warnings

logger = logging.getLogger('memory_monitor')

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_usage: float = 0.8):
        """
        初始化内存管理器
        
        Args:
            max_memory_usage: 最大内存使用率
        """
        self.max_memory_usage = max_memory_usage
        self.memory_threshold = max_memory_usage * psutil.virtual_memory().total
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存使用信息"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'free': memory.free
        }
    
    def check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        memory_info = self.get_memory_info()
        current_usage = memory_info['used'] / memory_info['total']
        
        logger.info(f"当前内存使用率: {current_usage:.2%}")
        
        if current_usage > self.max_memory_usage:
            logger.warning(f"内存使用率过高: {current_usage:.2%}")
            return False
        return True
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化DataFrame内存使用
        
        Args:
            df: 原始DataFrame
            
        Returns:
            pd.DataFrame: 优化后的DataFrame
        """
        original_memory = df.memory_usage(deep=True).sum()
        
        # 优化数值类型
        for col in df.columns:
            if df[col].dtype == 'object':
                # 尝试转换为category类型
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            
            elif df[col].dtype == 'int64':
                # 尝试使用更小的整数类型
                col_min = df[col].min()
                col_max = df[col].max()
                
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            elif df[col].dtype == 'float64':
                # 尝试使用float32
                if df[col].notna().all():
                    df[col] = df[col].astype(np.float32)
        
        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory
        
        logger.info(f"DataFrame内存优化: {original_memory/1024/1024:.2f}MB -> {optimized_memory/1024/1024:.2f}MB (减少 {reduction:.2%})")
        
        return df
    
    def force_garbage_collection(self):
        """强制垃圾回收"""
        collected = gc.collect()
        logger.info(f"垃圾回收完成，释放了 {collected} 个对象")
    
    def chunk_processing_wrapper(self, func, *args, chunk_size: int = 10000, **kwargs):
        """
        分块处理包装器
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            chunk_size: 分块大小
            **kwargs: 函数关键字参数
            
        Returns:
            处理结果
        """
        def process_chunk(chunk):
            return func(chunk, *args, **kwargs)
        
        return process_chunk

class DataFrameChunker:
    """DataFrame分块处理器"""
    
    def __init__(self, chunk_size: int = 10000):
        """
        初始化分块处理器
        
        Args:
            chunk_size: 分块大小
        """
        self.chunk_size = chunk_size
    
    def process_in_chunks(
        self, 
        df: pd.DataFrame, 
        process_func,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        分块处理DataFrame
        
        Args:
            df: 要处理的DataFrame
            process_func: 处理函数
            *args: 处理函数参数
            **kwargs: 处理函数关键字参数
            
        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        total_rows = len(df)
        chunks = []
        
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx].copy()
            
            logger.info(f"处理分块 {start_idx//self.chunk_size + 1}/{(total_rows-1)//self.chunk_size + 1}")
            
            # 处理分块
            processed_chunk = process_func(chunk, *args, **kwargs)
            chunks.append(processed_chunk)
            
            # 强制垃圾回收
            del chunk
            gc.collect()
        
        # 合并结果
        result = pd.concat(chunks, ignore_index=True)
        logger.info(f"分块处理完成，共处理 {len(result)} 行数据")
        
        return result
    
    def iterate_chunks(self, df: pd.DataFrame):
        """迭代DataFrame分块"""
        total_rows = len(df)
        
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx].copy()
            
            yield chunk
            
            # 清理内存
            del chunk
            gc.collect()

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, alert_threshold: float = 0.9):
        """
        初始化内存监控器
        
        Args:
            alert_threshold: 告警阈值
        """
        self.alert_threshold = alert_threshold
        self.memory_history = []
    
    def monitor_memory(self) -> Dict[str, Any]:
        """监控内存使用情况"""
        memory_info = psutil.virtual_memory()
        cpu_info = psutil.cpu_percent(interval=1)
        
        current_usage = {
            'timestamp': pd.Timestamp.now(),
            'memory_percent': memory_info.percent,
            'memory_used_gb': memory_info.used / 1024**3,
            'memory_available_gb': memory_info.available / 1024**3,
            'cpu_percent': cpu_info
        }
        
        self.memory_history.append(current_usage)
        
        # 检查告警
        if memory_info.percent > self.alert_threshold * 100:
            logger.warning(f"内存使用率告警: {memory_info.percent:.1f}%")
        
        return current_usage
    
    def get_memory_trend(self, hours: int = 1) -> pd.DataFrame:
        """获取内存使用趋势"""
        if not self.memory_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.memory_history)
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        
        return df[df['timestamp'] > cutoff_time]
    
    def generate_memory_report(self) -> str:
        """生成内存使用报告"""
        if not self.memory_history:
            return "无内存使用记录"
        
        df = pd.DataFrame(self.memory_history)
        
        report = f"""
内存使用报告:
- 平均内存使用率: {df['memory_percent'].mean():.1f}%
- 最大内存使用率: {df['memory_percent'].max():.1f}%
- 平均CPU使用率: {df['cpu_percent'].mean():.1f}%
- 监控时长: {len(df)} 个数据点
        """
        
        return report

def optimize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    优化数值列的数据类型
    
    Args:
        df: 原始DataFrame
        
    Returns:
        pd.DataFrame: 优化后的DataFrame
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        col_type = df[col].dtype
        
        if col_type == 'int64':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        
        elif col_type == 'float64':
            df[col] = df[col].astype(np.float32)
    
    return df

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    减少DataFrame内存使用
    
    Args:
        df: 原始DataFrame
        verbose: 是否输出详细信息
        
    Returns:
        pd.DataFrame: 优化后的DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'内存使用: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'内存使用: {end_mem:.2f} MB')
        print(f'减少: {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df 