# SSH隧道连接使用指南

## 概述

本系统支持通过SSH隧道连接远程数据库，确保数据传输的安全性和稳定性。通过SSH隧道，可以在本地安全地访问位于远程服务器上的数据库。

## 服务器配置

当前配置的服务器信息：
- **IP地址**: 39.96.116.223
- **用户名**: zlt
- **密码**: 123QWEasd03
- **SSH端口**: 22

## 数据库配置

通过SSH隧道连接的数据库信息：
- **远程主机**: 172.18.122.83:3306
- **本地端口**: localhost:3307
- **数据库**: gildata
- **用户**: jydb_ro

## 使用方法

### 1. 测试连接

在运行主程序之前，建议先测试连接：

```bash
# 测试SSH隧道连接
python scripts/test_ssh_connection.py

# 或者使用主程序的测试功能
python src/main.py --test-connection
```

### 2. 使用SSH隧道处理数据

```bash
# 使用SSH隧道连接处理股票数据
python src/main.py --use-ssh-tunnel process \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --include-technical \
    --include-features \
    --feature-types price volume technical
```

### 3. 在代码中使用SSH隧道

```python
from database.connector import JuyuanDB

# 使用SSH隧道连接
with JuyuanDB(use_ssh_tunnel=True) as db:
    # 执行查询
    result = db.query("SELECT * FROM your_table LIMIT 10")
    print(result)
```

### 4. 测试SSH隧道功能

```python
from utils.ssh_tunnel import test_ssh_connection, test_tunnel_connection
from config.settings import SSH_TUNNEL_CONFIG

# 测试SSH连接
if test_ssh_connection(SSH_TUNNEL_CONFIG):
    print("SSH连接成功")

# 测试隧道连接
if test_tunnel_connection(SSH_TUNNEL_CONFIG):
    print("隧道连接成功")
```

## 配置说明

### SSH隧道配置 (config/settings.py)

```python
SSH_TUNNEL_CONFIG = {
    'enabled': True,  # 启用SSH隧道
    'ssh_host': '39.96.116.223',
    'ssh_port': 22,
    'ssh_user': 'zlt',
    'ssh_password': '123QWEasd03',
    'local_port': 3307,  # 本地转发端口
    'remote_host': '172.18.122.83',  # 远程数据库主机
    'remote_port': 3306,  # 远程数据库端口
    'timeout': 30
}
```

### 数据库配置

```python
# 通过SSH隧道连接的数据库配置
SSH_DATABASE_CONFIG = {
    'host': 'localhost',  # 通过SSH隧道连接时使用localhost
    'port': 3307,  # 本地转发端口
    'user': 'jydb_ro',
    'password': '1qaz@WSX',
    'database': 'gildata',
    'charset': 'utf8mb4'
}
```

## 故障排除

### 1. SSH连接失败

**可能原因**：
- 网络连接问题
- 服务器IP或端口错误
- 用户名或密码错误
- 服务器SSH服务未启动

**解决方法**：
```bash
# 测试SSH连接
ssh zlt@39.96.116.223

# 检查网络连通性
ping 39.96.116.223

# 检查SSH端口
telnet 39.96.116.223 22
```

### 2. 隧道建立失败

**可能原因**：
- 本地端口被占用
- 远程数据库服务不可达
- SSH隧道配置错误

**解决方法**：
```python
# 检查本地端口占用
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 3307))
if result == 0:
    print("端口3307已被占用")
sock.close()

# 修改本地端口
SSH_TUNNEL_CONFIG['local_port'] = 3308
```

### 3. 数据库连接失败

**可能原因**：
- 数据库服务未启动
- 数据库用户权限不足
- 数据库配置错误

**解决方法**：
```python
# 测试数据库连接
from database.connector import test_database_connection

if test_database_connection(use_ssh_tunnel=True):
    print("数据库连接成功")
else:
    print("数据库连接失败")
```

## 性能优化

### 1. 连接池配置

```python
# 在数据库连接器中配置连接池
self.engine = create_engine(
    connection_string,
    pool_size=5,           # 连接池大小
    max_overflow=10,       # 最大溢出连接数
    pool_timeout=30,       # 连接超时时间
    pool_recycle=3600      # 连接回收时间
)
```

### 2. 超时设置

```python
# SSH连接超时
SSH_TUNNEL_CONFIG['timeout'] = 30

# 数据库连接超时
connect_timeout=30,
read_timeout=60
```

### 3. 连接复用

```python
# 使用单例模式管理SSH隧道
from utils.ssh_tunnel import SSHTunnelManager

tunnel_manager = SSHTunnelManager(SSH_TUNNEL_CONFIG)
# 在程序运行期间复用同一个隧道连接
```

## 安全注意事项

1. **密码安全**：避免在代码中硬编码密码，建议使用环境变量或配置文件
2. **网络安全**：确保SSH连接使用加密传输
3. **权限控制**：使用最小权限原则配置数据库用户
4. **连接监控**：定期检查连接状态和日志

## 监控和维护

### 1. 连接状态监控

```python
from utils.ssh_tunnel import SSHTunnelManager

tunnel_manager = SSHTunnelManager(SSH_TUNNEL_CONFIG)
if tunnel_manager.is_tunnel_active():
    print("SSH隧道活跃")
else:
    print("SSH隧道断开")
```

### 2. 日志监控

```python
import logging

# 设置SSH隧道日志级别
logging.getLogger('paramiko').setLevel(logging.INFO)
logging.getLogger('database').setLevel(logging.INFO)
```

### 3. 自动重连

```python
# 在数据库连接器中实现自动重连机制
def reconnect(self):
    """重新连接数据库"""
    self.close()
    time.sleep(5)  # 等待5秒
    self._setup_connection()
```

## 示例脚本

### 完整的特征提取示例

```python
#!/usr/bin/env python3
"""
使用SSH隧道进行特征提取的完整示例
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connector import JuyuanDB
from processing.feature_engine.price_features import QuantFeatureEngine
import pandas as pd

def main():
    # 使用SSH隧道连接数据库
    with JuyuanDB(use_ssh_tunnel=True) as db:
        # 查询股票数据
        sql = """
        SELECT 
            a.TradingDate,
            a.InnerCode,
            a.ClosePrice,
            a.TurnoverVolume,
            b.MACD,
            c.RSI
        FROM 
            `境内股票复权行情表` a
        LEFT JOIN `境内股票强弱与趋向技术指标` b 
            ON a.InnerCode = b.InnerCode AND a.TradingDate = b.TradingDate
        LEFT JOIN `境内股票摆动与反趋向技术指标` c 
            ON a.InnerCode = c.InnerCode AND a.TradingDate = c.TradingDate
        WHERE 
            a.TradingDate BETWEEN '2023-12-01' AND '2023-12-31'
            AND a.ClosePrice > 0
        LIMIT 1000
        """
        
        # 读取数据
        df = db.read_sql(sql)
        print(f"读取到 {len(df)} 条记录")
        
        # 计算特征
        feature_engine = QuantFeatureEngine()
        df_with_features = feature_engine.calculate_all_features(df)
        
        # 保存结果
        df_with_features.to_feather('output/features_with_ssh.feather')
        print("特征计算完成，结果已保存")

if __name__ == "__main__":
    main()
```

## 总结

SSH隧道连接为量化特征系统提供了安全、稳定的数据库访问方式。通过合理配置和监控，可以确保系统的可靠性和性能。建议在生产环境中定期测试连接状态，并建立相应的监控和告警机制。 