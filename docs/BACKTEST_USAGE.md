# 回测系统使用说明

## 概述

量化特征系统现在支持完整的回测功能，可以设置不同的时间范围进行策略回测。

## 快速开始

### 1. 查看可用的回测配置

```bash
# 查看所有回测周期
python scripts/run_backtest.py --list-periods

# 查看所有参数配置
python scripts/run_backtest.py --list-params
```

### 2. 运行标准回测

```bash
# 使用默认配置运行2024年全年回测
python scripts/run_backtest.py --period medium_term --params base

# 使用SSH隧道连接
python scripts/run_backtest.py --period medium_term --params base --use-ssh-tunnel
```

### 3. 运行快速测试

```bash
# 运行短期回测（2024年第一季度）
python scripts/run_backtest.py --period short_term --params quick_test
```

## 回测周期配置

| 周期名称 | 时间范围 | 描述 |
|---------|---------|------|
| `short_term` | 2024-01-01 到 2024-03-31 | 短期回测，用于快速验证策略 |
| `medium_term` | 2024-01-01 到 2024-12-31 | 中期回测，标准回测周期 |
| `long_term` | 2023-01-01 到 2024-12-31 | 长期回测，历史验证 |
| `bull_market` | 2020-03-01 到 2021-12-31 | 牛市回测 |
| `bear_market` | 2022-01-01 到 2022-12-31 | 熊市回测 |
| `sideways_market` | 2023-01-01 到 2023-12-31 | 震荡市回测 |
| `recent` | 2024-10-01 到 2024-12-31 | 最近数据回测 |

## 参数配置

| 配置名称 | 描述 | 特点 |
|---------|------|------|
| `base` | 基础配置 | 包含技术指标和量化特征，前复权 |
| `quick_test` | 快速测试 | 不包含技术指标和特征，用于快速验证 |
| `full_features` | 完整特征 | 包含所有特征类型，最全面的分析 |

## 使用主程序进行回测

### 1. 直接指定时间范围

```bash
# 使用主程序指定回测时间范围
python src/main.py process \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --include-technical \
    --include-features \
    --feature-types price volume technical
```

### 2. 测试模式

```bash
# 测试模式（只处理最近一周数据）
python src/main.py process --test-mode
```

## 回测输出

### 1. 数据文件

回测结果会保存在以下位置：
- `output/processed_data/` - 处理后的数据文件
- `output/reports/` - 分析报告
- `output/logs/` - 日志文件

### 2. 报告内容

每次回测会生成以下报告：
- **数据报告**: 数据统计、质量分析
- **特征报告**: 特征计算、相关性分析
- **内存报告**: 系统性能监控

## 高级用法

### 1. 自定义回测配置

可以修改 `config/backtest_config.py` 文件来添加自定义的回测配置：

```python
# 添加自定义回测周期
BACKTEST_PERIODS['custom'] = {
    'name': '自定义回测',
    'start_date': '2023-06-01',
    'end_date': '2024-05-31',
    'description': '自定义时间范围'
}

# 添加自定义参数配置
BACKTEST_PARAMS['custom'] = {
    'market_codes': [83],  # 只包含深交所
    'adj_type': 'backward',  # 后复权
    'include_technical': True,
    'include_features': True,
    'feature_types': ['price', 'volume'],
    'output_format': 'csv'
}
```

### 2. 批量回测

可以编写脚本进行批量回测：

```python
from scripts.run_backtest import run_backtest

# 批量运行不同周期的回测
periods = ['short_term', 'medium_term', 'long_term']
params = ['base', 'full_features']

for period in periods:
    for param in params:
        print(f"运行回测: {period} + {param}")
        success = run_backtest(period, param)
        if success:
            print(f"✅ {period} + {param} 完成")
        else:
            print(f"❌ {period} + {param} 失败")
```

### 3. 回测结果分析

```python
import pandas as pd
from pathlib import Path

# 读取回测结果
data_file = Path('output/processed_data/stock_data_20241201_120000.parquet')
df = pd.read_parquet(data_file)

# 分析回测结果
print(f"数据时间范围: {df['TradingDay'].min()} 到 {df['TradingDay'].max()}")
print(f"股票数量: {df['SecuCode'].nunique()}")
print(f"特征数量: {len([col for col in df.columns if col not in ['SecuCode', 'TradingDay']])}")
```

## 注意事项

1. **数据量**: 长期回测可能涉及大量数据，注意内存使用
2. **计算时间**: 包含技术指标和特征的回测需要较长计算时间
3. **存储空间**: 完整特征的回测结果文件较大
4. **数据库连接**: 确保数据库连接稳定，长时间查询可能需要SSH隧道

## 故障排除

### 1. 内存不足

```bash
# 使用快速测试配置
python scripts/run_backtest.py --period short_term --params quick_test
```

### 2. 数据库连接超时

```bash
# 使用SSH隧道
python scripts/run_backtest.py --period short_term --params base --use-ssh-tunnel
```

### 3. 查看详细日志

日志文件保存在 `output/logs/` 目录下，可以查看详细的执行信息。

## 示例场景

### 场景1: 策略快速验证

```bash
# 使用短期回测快速验证策略
python scripts/run_backtest.py --period short_term --params base
```

### 场景2: 完整策略回测

```bash
# 使用完整特征进行长期回测
python scripts/run_backtest.py --period long_term --params full_features
```

### 场景3: 市场环境测试

```bash
# 测试牛市环境
python scripts/run_backtest.py --period bull_market --params base

# 测试熊市环境
python scripts/run_backtest.py --period bear_market --params base
``` 