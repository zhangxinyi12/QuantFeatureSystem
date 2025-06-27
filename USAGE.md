# 量化特征系统使用说明

## 概述

量化特征系统是一个统一的股票数据处理、特征工程、监控和分析系统。系统提供了多种功能模块，支持从数据库查询数据、生成量化特征、监控系统性能、验证数据质量等。

## 主要功能

### 1. 数据处理 (process)
从数据库查询股票数据并进行处理，包括：
- 价格复权
- 技术指标计算
- 量化特征生成
- 数据保存

### 2. 系统监控 (monitor)
监控系统性能和资源使用情况

### 3. 数据验证 (validate)
验证数据质量和完整性

### 4. 特征生成 (features)
从现有数据文件生成量化特征

### 5. 报告生成 (report)
生成数据分析报告和可视化

### 6. 漂移检测 (drift)
检测数据分布漂移

### 7. 内存优化 (optimize)
优化内存使用

### 8. 系统信息 (info)
显示系统基本信息

## 使用方法

### 基本语法
```bash
python main.py <command> [options]
```

### 命令示例

#### 1. 处理股票数据
```bash
# 基本数据处理
python main.py process --start-date 2024-01-01 --end-date 2024-12-31

# 包含技术指标和量化特征
python main.py process --start-date 2024-01-01 --end-date 2024-12-31 \
    --include-technical --include-features \
    --feature-types price volume technical

# 测试模式（只处理最近一周数据）
python main.py process --test-mode --include-features

# 指定输出格式
python main.py process --output-format parquet --include-features
```

#### 2. 系统监控
```bash
# 监控60分钟
python main.py monitor

# 监控指定时间
python main.py monitor --duration 120
```

#### 3. 数据验证
```bash
# 验证数据文件
python main.py validate --data data.csv

# 验证指定列
python main.py validate --data data.csv --columns date open high low close volume
```

#### 4. 特征生成
```bash
# 从文件生成特征
python main.py features --data input.csv

# 指定特征类型和输出路径
python main.py features --data input.csv \
    --feature-types volume price technical \
    --output output/features.csv
```

#### 5. 报告生成
```bash
# 生成分析报告
python main.py report --data data.csv

# 指定输出目录
python main.py report --data data.csv --output reports/
```

#### 6. 漂移检测
```bash
# 检测数据漂移
python main.py drift --reference reference.csv --current current.csv
```

#### 7. 内存优化
```bash
# 优化内存使用
python main.py optimize
```

#### 8. 系统信息
```bash
# 显示系统信息
python main.py info
```

## 参数说明

### 数据处理参数
- `--start-date`: 开始日期 (YYYY-MM-DD)
- `--end-date`: 结束日期 (YYYY-MM-DD)
- `--market-codes`: 市场代码列表
- `--output-format`: 输出格式 (feather/parquet/csv)
- `--chunk-size`: 分块大小
- `--adj-type`: 复权类型 (forward/backward/none)
- `--include-technical`: 是否计算技术指标
- `--include-features`: 是否计算量化特征
- `--feature-types`: 特征类型列表
- `--test-mode`: 测试模式

### 特征类型选项
- `price`: 价格特征
- `volume`: 成交量特征
- `technical`: 技术指标
- `order_book`: 订单簿特征
- `fundamental`: 基本面特征
- `alternative`: 另类统计特征
- `cross_asset`: 跨资产特征

## 输出文件

### 数据文件
- 处理后的数据保存在 `output/processed_data/` 目录
- 文件名格式：`stock_data_YYYYMMDD_HHMMSS.{format}`

### 报告文件
- 数据报告保存在 `output/reports/` 目录
- 文件名格式：`data_report_YYYYMMDD_HHMMSS.txt`
- 特征报告：`feature_report_YYYYMMDD_HHMMSS.txt`

### 日志文件
- 日志文件保存在 `logs/` 目录
- 文件名格式：`quant_feature_system_YYYYMMDD_HHMMSS.log`

## 配置说明

系统会自动尝试加载 `config/settings.py` 中的配置。如果配置文件不存在，将使用默认配置：

```python
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
```

## 错误处理

系统具有完善的错误处理机制：
- 模块导入失败时会显示警告但继续运行
- 数据库连接失败会记录错误并退出
- 数据处理过程中的异常会被捕获并记录
- 用户中断（Ctrl+C）会被正确处理

## 性能优化

- 支持数据分块处理，避免内存溢出
- 自动内存优化和垃圾回收
- 可配置的监控和性能报告
- 支持多种数据格式以平衡存储和读取性能

## 注意事项

1. 首次运行前请确保数据库连接配置正确
2. 大量数据处理时建议使用测试模式先验证
3. 特征生成可能需要较长时间，请耐心等待
4. 建议定期清理日志文件以节省磁盘空间
5. 监控功能会持续运行，请根据需要设置合适的持续时间 