#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据存储策略配置
定义数据存储的格式、分区、压缩等策略
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class StorageStrategy:
    """存储策略配置"""
    
    # 基础配置
    format: str = 'parquet'  # 存储格式: parquet, csv, hdf5
    compression: str = 'snappy'  # 压缩方式: snappy, gzip, lz4
    encoding: str = 'utf-8'  # 编码格式
    
    # 分区配置
    partition_by: List[str] = None  # 分区字段
    partition_size: int = 1000000  # 分区大小（行数）
    
    # 性能配置
    chunk_size: int = 10000  # 写入块大小
    max_file_size: str = '1GB'  # 最大文件大小
    
    # 元数据配置
    include_metadata: bool = True  # 是否包含元数据
    metadata_format: str = 'json'  # 元数据格式
    
    def __post_init__(self):
        if self.partition_by is None:
            self.partition_by = ['date', 'market_code']

class StorageConfig:
    """存储配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化存储配置
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.strategies = self._load_default_strategies()
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
    
    def _load_default_strategies(self) -> Dict[str, StorageStrategy]:
        """加载默认存储策略"""
        return {
            'default': StorageStrategy(),
            'high_performance': StorageStrategy(
                format='parquet',
                compression='lz4',
                chunk_size=50000,
                partition_size=500000
            ),
            'high_compression': StorageStrategy(
                format='parquet',
                compression='gzip',
                chunk_size=20000,
                partition_size=2000000
            ),
            'csv_export': StorageStrategy(
                format='csv',
                compression='gzip',
                encoding='utf-8',
                include_metadata=False
            ),
            'feature_data': StorageStrategy(
                format='parquet',
                compression='snappy',
                partition_by=['date', 'feature_type'],
                partition_size=500000,
                include_metadata=True
            ),
            'raw_data': StorageStrategy(
                format='parquet',
                compression='snappy',
                partition_by=['date', 'market_code', 'data_type'],
                partition_size=1000000,
                include_metadata=True
            )
        }
    
    def get_strategy(self, name: str) -> StorageStrategy:
        """
        获取存储策略
        
        Args:
            name: 策略名称
            
        Returns:
            存储策略对象
        """
        return self.strategies.get(name, self.strategies['default'])
    
    def add_strategy(self, name: str, strategy: StorageStrategy):
        """
        添加自定义存储策略
        
        Args:
            name: 策略名称
            strategy: 存储策略对象
        """
        self.strategies[name] = strategy
    
    def save_config(self, config_path: str = None):
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = self.config_path
            
        if config_path is None:
            raise ValueError("配置文件路径未指定")
        
        # 转换策略为可序列化的字典
        config_data = {}
        for name, strategy in self.strategies.items():
            config_data[name] = {
                'format': strategy.format,
                'compression': strategy.compression,
                'encoding': strategy.encoding,
                'partition_by': strategy.partition_by,
                'partition_size': strategy.partition_size,
                'chunk_size': strategy.chunk_size,
                'max_file_size': strategy.max_file_size,
                'include_metadata': strategy.include_metadata,
                'metadata_format': strategy.metadata_format
            }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def load_config(self, config_path: str):
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        for name, data in config_data.items():
            strategy = StorageStrategy(
                format=data.get('format', 'parquet'),
                compression=data.get('compression', 'snappy'),
                encoding=data.get('encoding', 'utf-8'),
                partition_by=data.get('partition_by', ['date', 'market_code']),
                partition_size=data.get('partition_size', 1000000),
                chunk_size=data.get('chunk_size', 10000),
                max_file_size=data.get('max_file_size', '1GB'),
                include_metadata=data.get('include_metadata', True),
                metadata_format=data.get('metadata_format', 'json')
            )
            self.strategies[name] = strategy
    
    def get_formats_info(self) -> Dict[str, Dict]:
        """获取支持的格式信息"""
        return {
            'parquet': {
                'description': '列式存储格式，压缩率高，查询性能好',
                'compression': ['snappy', 'gzip', 'lz4', 'brotli'],
                'advantages': ['高压缩率', '快速查询', '列式存储'],
                'disadvantages': ['写入较慢', '需要额外库']
            },
            'csv': {
                'description': '文本格式，兼容性好',
                'compression': ['gzip', 'bz2', 'xz'],
                'advantages': ['兼容性好', '易于查看', '标准格式'],
                'disadvantages': ['压缩率低', '查询慢', '无类型信息']
            },
            'hdf5': {
                'description': '科学计算格式，支持复杂数据结构',
                'compression': ['gzip', 'lzf', 'szip'],
                'advantages': ['支持复杂数据', '快速I/O', '元数据丰富'],
                'disadvantages': ['文件大小限制', '依赖HDF5库']
            }
        }
    
    def recommend_strategy(self, data_size: int, query_pattern: str = 'random') -> str:
        """
        根据数据特征推荐存储策略
        
        Args:
            data_size: 数据大小（行数）
            query_pattern: 查询模式 ('random', 'sequential', 'column')
            
        Returns:
            推荐的策略名称
        """
        if data_size < 100000:
            return 'default'
        elif data_size < 1000000:
            if query_pattern == 'column':
                return 'feature_data'
            else:
                return 'high_performance'
        else:
            if query_pattern == 'column':
                return 'feature_data'
            else:
                return 'high_compression' 