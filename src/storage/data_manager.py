#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据存储管理器
整合分区管理和存储策略，提供统一的数据存储接口
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import json

from .partition_manager import PartitionManager
from .storage_config import StorageConfig, StorageStrategy

logger = logging.getLogger(__name__)

class DataManager:
    """数据存储管理器"""
    
    def __init__(self, base_path: str = "output/processed_data", 
                 config_path: Optional[str] = None):
        """
        初始化数据管理器
        
        Args:
            base_path: 数据存储基础路径
            config_path: 存储配置文件路径
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.partition_manager = PartitionManager()
        self.storage_config = StorageConfig(config_path)
        
        # 创建子目录
        self._create_subdirectories()
        
    def _create_subdirectories(self):
        """创建存储子目录"""
        subdirs = [
            'raw_data',
            'feature_data', 
            'processed_data',
            'temporary',
            'backup',
            'metadata'
        ]
        
        for subdir in subdirs:
            (self.base_path / subdir).mkdir(exist_ok=True)
    
    def save_data(self, df: pd.DataFrame, 
                  data_type: str = 'processed',
                  strategy_name: str = 'default',
                  partition_by: Optional[List[str]] = None,
                  metadata: Optional[Dict] = None) -> str:
        """
        保存数据
        
        Args:
            df: 要保存的数据
            data_type: 数据类型 ('raw', 'feature', 'processed')
            strategy_name: 存储策略名称
            partition_by: 分区字段列表
            metadata: 元数据
            
        Returns:
            保存的文件路径
        """
        logger.info(f"开始保存数据: {data_type}, 策略: {strategy_name}")
        
        # 获取存储策略
        strategy = self.storage_config.get_strategy(strategy_name)
        
        # 确定保存路径
        save_path = self._get_save_path(data_type, strategy)
        
        # 添加元数据
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'created_at': datetime.now().isoformat(),
            'data_type': data_type,
            'strategy': strategy_name,
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum()
        })
        
        # 根据策略保存数据
        if partition_by and strategy.partition_by:
            file_path = self._save_partitioned(df, save_path, strategy, partition_by, metadata)
        else:
            file_path = self._save_single_file(df, save_path, strategy, metadata)
        
        logger.info(f"数据保存完成: {file_path}")
        return str(file_path)
    
    def _get_save_path(self, data_type: str, strategy: StorageStrategy) -> Path:
        """获取保存路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if data_type == 'raw':
            subdir = 'raw_data'
        elif data_type == 'feature':
            subdir = 'feature_data'
        else:
            subdir = 'processed_data'
        
        return self.base_path / subdir / f"{data_type}_{timestamp}"
    
    def _save_partitioned(self, df: pd.DataFrame, base_path: Path, 
                         strategy: StorageStrategy, partition_by: List[str],
                         metadata: Dict) -> Path:
        """保存分区数据"""
        # 使用分区管理器
        partition_info = self.partition_manager.create_partitions(
            df, partition_by, strategy.partition_size
        )
        
        # 保存分区数据
        for partition_name, partition_data in partition_info['partitions'].items():
            partition_path = base_path / partition_name
            partition_path.mkdir(parents=True, exist_ok=True)
            
            file_name = f"data.{strategy.format}"
            file_path = partition_path / file_name
            
            self._write_file(partition_data, file_path, strategy)
        
        # 保存分区元数据
        metadata_path = base_path / 'partition_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(partition_info, f, indent=2, ensure_ascii=False)
        
        # 保存数据元数据
        if strategy.include_metadata:
            data_metadata_path = base_path / 'data_metadata.json'
            with open(data_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return base_path
    
    def _save_single_file(self, df: pd.DataFrame, base_path: Path,
                         strategy: StorageStrategy, metadata: Dict) -> Path:
        """保存单文件数据"""
        file_name = f"data.{strategy.format}"
        file_path = base_path.with_suffix(f'.{strategy.format}')
        
        # 写入数据文件
        self._write_file(df, file_path, strategy)
        
        # 保存元数据
        if strategy.include_metadata:
            metadata_path = file_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return file_path
    
    def _write_file(self, df: pd.DataFrame, file_path: Path, strategy: StorageStrategy):
        """写入文件"""
        if strategy.format == 'parquet':
            df.to_parquet(
                file_path,
                compression=strategy.compression,
                index=False
            )
        elif strategy.format == 'csv':
            df.to_csv(
                file_path,
                compression=strategy.compression,
                encoding=strategy.encoding,
                index=False
            )
        elif strategy.format == 'hdf5':
            df.to_hdf(
                file_path,
                key='data',
                mode='w',
                complevel=9 if strategy.compression == 'gzip' else 0
            )
        else:
            raise ValueError(f"不支持的格式: {strategy.format}")
    
    def load_data(self, file_path: str, 
                  chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            file_path: 文件路径
            chunk_size: 分块大小（用于大文件）
            
        Returns:
            加载的数据
        """
        logger.info(f"开始加载数据: {file_path}")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 检查是否为分区数据
        if file_path.is_dir():
            return self._load_partitioned_data(file_path, chunk_size)
        else:
            return self._load_single_file(file_path, chunk_size)
    
    def _load_partitioned_data(self, dir_path: Path, chunk_size: Optional[int]) -> pd.DataFrame:
        """加载分区数据"""
        # 读取分区元数据
        metadata_path = dir_path / 'partition_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                partition_info = json.load(f)
        else:
            # 如果没有元数据，扫描目录
            partition_info = {'partitions': {}}
            for subdir in dir_path.iterdir():
                if subdir.is_dir():
                    partition_info['partitions'][subdir.name] = str(subdir)
        
        # 加载所有分区数据
        dfs = []
        for partition_name, partition_path in partition_info['partitions'].items():
            partition_path = Path(partition_path)
            
            # 查找数据文件
            for file_path in partition_path.glob('*'):
                if file_path.suffix in ['.parquet', '.csv', '.h5', '.hdf5']:
                    df = self._load_single_file(file_path, chunk_size)
                    dfs.append(df)
                    break
        
        if not dfs:
            raise ValueError(f"未找到分区数据文件: {dir_path}")
        
        # 合并数据
        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"分区数据加载完成: {len(result)} 行")
        return result
    
    def _load_single_file(self, file_path: Path, chunk_size: Optional[int]) -> pd.DataFrame:
        """加载单文件数据"""
        if file_path.suffix == '.parquet':
            if chunk_size:
                chunks = []
                for chunk in pd.read_parquet(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
            else:
                return pd.read_parquet(file_path)
        
        elif file_path.suffix == '.csv':
            if chunk_size:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
            else:
                return pd.read_csv(file_path)
        
        elif file_path.suffix in ['.h5', '.hdf5']:
            return pd.read_hdf(file_path, key='data')
        
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        info = {
            'base_path': str(self.base_path),
            'total_size': 0,
            'file_count': 0,
            'data_types': {},
            'strategies': list(self.storage_config.strategies.keys())
        }
        
        # 统计存储信息
        for file_path in self.base_path.rglob('*'):
            if file_path.is_file():
                info['file_count'] += 1
                info['total_size'] += file_path.stat().st_size
                
                # 统计数据类型
                if 'raw_data' in str(file_path):
                    info['data_types']['raw'] = info['data_types'].get('raw', 0) + 1
                elif 'feature_data' in str(file_path):
                    info['data_types']['feature'] = info['data_types'].get('feature', 0) + 1
                elif 'processed_data' in str(file_path):
                    info['data_types']['processed'] = info['data_types'].get('processed', 0) + 1
        
        return info
    
    def cleanup_temporary(self, max_age_days: int = 7):
        """清理临时文件"""
        temp_dir = self.base_path / 'temporary'
        if not temp_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.info(f"删除临时文件: {file_path}")
    
    def backup_data(self, backup_name: str = None) -> str:
        """备份数据"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.base_path / 'backup' / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 复制数据文件
        import shutil
        for subdir in ['raw_data', 'feature_data', 'processed_data']:
            src_dir = self.base_path / subdir
            if src_dir.exists():
                dst_dir = backup_path / subdir
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        
        logger.info(f"数据备份完成: {backup_path}")
        return str(backup_path) 