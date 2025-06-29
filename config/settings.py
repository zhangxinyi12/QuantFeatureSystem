"""
项目配置文件
包含数据库连接、数据处理参数等配置
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 服务器配置
SERVER_CONFIG = {
    'host': '39.96.116.223',
    'username': 'zlt',
    'password': '123QWEasd03',
    'port': 22,  # SSH端口
    'timeout': 30
}

# 数据库配置（直接连接）
DATABASE_CONFIG = {
    'host': '172.18.122.83',  # 直接使用数据库服务器IP
    'port': 3306,
    'user': 'jydb_ro',
    'password': '1qaz@WSX',
    'database': 'gildata',
    'charset': 'utf8mb4'
}

# SSH隧道配置（禁用）
SSH_TUNNEL_CONFIG = {
    'enabled': False,  # 禁用SSH隧道
    'ssh_host': '39.96.116.223',
    'ssh_port': 22,
    'ssh_user': 'zlt',
    'ssh_password': '123QWEasd03',
    'local_port': 3307,
    'remote_host': '172.18.122.83',
    'remote_port': 3306,
    'timeout': 30
}

# 通过SSH隧道连接的数据库配置（不再使用）
SSH_DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'jydb_ro',
    'password': '1qaz@WSX',
    'database': 'gildata',
    'charset': 'utf8mb4'
}

# 数据处理配置
DATA_CONFIG = {
    # 默认查询时间范围
    'default_start_date': '2015-01-01',
    'default_end_date': '2024-12-31',
    
    # 批量处理配置
    'batch_size': 1000,  # 每批处理的记录数
    'chunk_size': 50000,  # pandas读取时的分块大小
    
    # 股票市场代码
    'market_codes': [83, 90],  # 沪深A股市场
    
    # 数据保存格式
    'output_format': 'feather',  # feather, parquet, csv
    'compression': 'snappy'  # 压缩格式
}

# 内存优化配置
MEMORY_CONFIG = {
    'max_memory_usage': 0.8,  # 最大内存使用率
    'gc_threshold': 0.7,      # 垃圾回收阈值
    'chunk_processing': True,  # 启用分块处理
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': PROJECT_ROOT / 'output' / 'logs' / 'app.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# 输出目录配置
OUTPUT_CONFIG = {
    'base_dir': PROJECT_ROOT / 'output',
    'processed_data_dir': PROJECT_ROOT / 'output' / 'processed_data',
    'reports_dir': PROJECT_ROOT / 'output' / 'reports',
    'logs_dir': PROJECT_ROOT / 'output' / 'logs'
}

# 确保输出目录存在
for dir_path in OUTPUT_CONFIG.values():
    if isinstance(dir_path, Path):
        dir_path.mkdir(parents=True, exist_ok=True) 