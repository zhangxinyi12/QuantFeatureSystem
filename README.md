# 量化特征系统 (QuantFeatureSystem)

一个专门用于从聚源数据库读取股票数据、处理时序数据并在低配服务器上高效运行的Python项目。

## 项目特点

- 🚀 **高效数据处理**: 针对低配服务器优化的内存管理和分块处理
- 📊 **完整时序处理**: 支持价格复权、技术指标计算、成交量分析
- 🔧 **模块化设计**: 清晰的代码结构，易于维护和扩展
- 📈 **聚源数据集成**: 专门针对聚源数据库优化的查询和连接
- 🛡️ **错误处理**: 完善的异常处理和日志记录
- 🎯 **量化特征工程**: 全面的价格和成交量衍生特征计算

## 项目结构

```
QuantFeatureSystem/
├── config/                  # 配置文件
│   ├── settings.py          # 数据库连接配置等
│   └── logging.conf         # 日志配置
├── data_schema/             # 数据字典和表结构
│   ├── juyuan_dictionary.md # 聚源数据字段说明
│   ├── feature_dictionary.md # 量化特征字典
│   └── table_schema.sql     # 相关表结构
├── src/
│   ├── database/            # 数据库操作
│   │   ├── connector.py     # 数据库连接
│   │   └── queries.py       # SQL查询语句
│   ├── processing/          # 数据处理
│   │   ├── timeseries.py    # 时序数据处理函数
│   │   └── feature_engine/  # 量化特征工程
│   │       ├── price.py     # 价格衍生特征
│   │       └── vol.py       # 成交量衍生特征
│   ├── utils/               # 工具函数
│   │   ├── memory_utils.py  # 内存优化工具
│   │   ├── logging_utils.py # 日志工具
│   │   └── monitor.py       # 资源监控
│   └── main.py              # 主程序入口
├── output/                  # 输出结果
│   ├── reports/             # 分析报告
│   └── processed_data/      # 处理后的数据
├── scripts/                 # 辅助脚本
├── examples/                # 使用示例
├── requirements.txt         # 依赖库
└── README.md                # 项目说明
```

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**重要依赖说明:**
- `TA-Lib>=0.4.24`: 技术分析库，用于计算MFI等指标
- `scipy>=1.9.0`: 科学计算库
- `matplotlib>=3.5.0`: 可视化库

### 2. 配置数据库连接

编辑 `config/settings.py` 文件，修改数据库连接参数：

```python
DATABASE_CONFIG = {
    'host': 'your_database_host',
    'port': 3306,
    'user': 'your_username',
    'password': 'your_password',
    'database': 'gildata',
    'charset': 'utf8mb4'
}
```

### 3. 创建输出目录

```bash
mkdir -p output/reports output/processed_data output/logs
```

## 使用方法

### 基本用法

```bash
# 查询2023年全年的股票数据
python src/main.py --start-date 2023-01-01 --end-date 2023-12-31

# 测试模式（只查询最近一周数据）
python src/main.py --test-mode

# 包含技术指标计算
python src/main.py --include-technical --adj-type forward

# 包含量化特征计算
python src/main.py --include-features --feature-types price volume technical

# 指定输出格式
python src/main.py --output-format parquet --chunk-size 10000
```

### 命令行参数

- `--start-date`: 开始日期 (YYYY-MM-DD)
- `--end-date`: 结束日期 (YYYY-MM-DD)
- `--market-codes`: 市场代码列表 (默认: 83, 90)
- `--output-format`: 输出格式 (feather/parquet/csv)
- `--chunk-size`: 分块大小 (默认: 50000)
- `--adj-type`: 复权类型 (forward/backward/none)
- `--include-technical`: 是否计算技术指标
- `--include-features`: 是否计算量化特征
- `--feature-types`: 要计算的特征类型 (price/volume/technical/cross_asset)
- `--test-mode`: 测试模式

## 功能特性

### 数据处理功能

1. **价格复权处理**
   - 前复权 (Forward Adjustment)
   - 后复权 (Backward Adjustment)
   - 不复权 (No Adjustment)

2. **技术指标计算**
   - 移动平均线 (MA5, MA10, MA20, MA60)
   - 相对强弱指数 (RSI)
   - MACD指标
   - 波动率计算
   - 支撑阻力位

3. **成交量分析**
   - 成交量移动平均
   - 成交量比率
   - 价量相关性

### 量化特征工程

#### 价格衍生特征

1. **收益率特征**
   - 绝对收益率 (1d, 5d, 20d, 60d, 252d)
   - 对数收益率
   - 累积收益率

2. **动量特征**
   - 简单动量
   - 价格变化率 (ROC)
   - 相对动量

3. **波动率特征**
   - 历史波动率
   - 已实现波动率
   - Parkinson波动率
   - Garman-Klass波动率
   - 平均真实波幅 (ATR)

4. **趋势指标**
   - 简单移动平均 (SMA)
   - 指数移动平均 (EMA)
   - 布林带
   - 价格相对位置

5. **支撑阻力位**
   - 近期支撑阻力位
   - 斐波那契回撤位
   - 价格位置指标

#### 成交量衍生特征

成交量衍生特征是量化交易中的重要组成部分，能够反映市场参与度和资金流向。本系统实现了完整的成交量特征工程模块。

##### 1. 绝对成交量特征

**当期成交量**
- `Volume_Current`: 当期成交量
- 用途: 反映当前交易活跃度

**平均成交量**
- `Volume_MA_5`, `Volume_MA_20`, `Volume_MA_50`, `Volume_MA_200`: 不同窗口的平均成交量
- 用途: 判断成交量相对于历史水平的活跃程度

```python
from processing.feature_engine.volume_features import calculate_absolute_volume

# 计算绝对成交量特征
df = calculate_absolute_volume(df, windows=[5, 20, 50, 200])
```

##### 2. 成交量变化特征

**成交量变化率**
- `Volume_Change_Rate_5`, `Volume_Change_Rate_10`, `Volume_Change_Rate_20`: 不同周期的成交量变化率
- 计算公式: `(当前成交量 - N日前成交量) / N日前成交量`
- 用途: 识别成交量异常变化

**成交量动量**
- `Volume_Momentum_5`, `Volume_Momentum_10`, `Volume_Momentum_20`: 成交量绝对变化量
- 计算公式: `当前成交量 - N日前成交量`
- 用途: 衡量成交量变化的绝对幅度

```python
from processing.feature_engine.volume_features import calculate_volume_change

# 计算成交量变化特征
df = calculate_volume_change(df, windows=[5, 10, 20])
```

##### 3. 量价结合特征

**累积/派发线 (A/D Line)**
- `AD_Line`: 累积资金流指标
- 计算公式: `CLV * Volume` 的累积和，其中 `CLV = ((Close - Low) - (High - Close)) / (High - Low)`
- 用途: 判断资金流入流出趋势

**资金流量指标 (MFI)**
- `MFI`: 资金流量指标，范围0-100
- 计算公式: 基于价格和成交量的震荡指标
- 用途: 识别超买超卖区域，判断价格趋势的可靠性

**量价背离检测**
- `Divergence_PriceHigh_VolumeLow`: 价格创新高但成交量未配合
- `Divergence_PriceLow_VolumeLow`: 价格创新低但成交量未配合
- `Divergence_PriceLow_VolumeHigh`: 价格未创新高但成交量创新高
- `Divergence_PriceHigh_VolumeLow`: 价格未创新低但成交量创新低
- 用途: 识别潜在的趋势转折点

**成交量加权平均价 (VWAP)**
- `VWAP_15`, `VWAP_30`, `VWAP_60`, `VWAP_240`: 不同时间窗口的VWAP
- 计算公式: `Σ(典型价格 × 成交量) / Σ成交量`
- 用途: 判断价格相对于平均成交成本的位置

```python
from processing.feature_engine.volume_features import (
    calculate_accumulation_distribution,
    calculate_mfi,
    detect_volume_price_divergence,
    calculate_vwap
)

# 计算量价结合特征
df = calculate_accumulation_distribution(df)
df = calculate_mfi(df, period=14)
df = detect_volume_price_divergence(df, window=20)
df = calculate_vwap(df, window=30)
```

##### 4. 订单流不平衡特征 (需要Level 2数据)

**订单流不平衡 (OFI)**
- `OFI`: 买卖订单流不平衡指标
- 计算公式: `(Bid_Size1 - Ask_Size1) / (Bid_Size1 + Ask_Size1)`
- 用途: 判断买卖压力平衡

**价格压力**
- `Price_Pressure`: 价格压力指标
- 计算公式: `(Ask_Price1 - Mid_Price) / (Ask_Price1 - Bid_Price1)`
- 用途: 衡量价格偏离中间价的程度

```python
from processing.feature_engine.volume_features import calculate_order_flow_imbalance

# 计算订单流不平衡特征 (需要Level 2数据)
df = calculate_order_flow_imbalance(level2_df)
```

##### 5. 换手率特征

**换手率**
- `Turnover_Rate`: 日换手率
- 计算公式: `成交量 / 流通股本`
- 用途: 衡量股票流动性，识别异常活跃的交易

```python
from processing.feature_engine.volume_features import calculate_turnover_rate

# 计算换手率 (需要提供流通股本)
float_shares = 100000000  # 1亿股流通股
df = calculate_turnover_rate(df, float_shares=float_shares)
```

##### 6. 大单/异常交易识别 (需要Tick数据)

**大单检测**
- `Block_Trade`: 大单标识 (0/1)
- 检测条件: 成交量 > 平均成交量 × 倍数阈值

**异常价格交易**
- `Abnormal_Price_Trade`: 异常价格交易标识 (0/1)
- 检测条件: 价格偏离典型价格超过阈值

**异常成交量交易**
- `Abnormal_Volume_Trade`: 异常成交量交易标识 (0/1)
- 检测条件: 成交量偏离均值超过3个标准差

```python
from processing.feature_engine.volume_features import detect_block_trades

# 检测大单和异常交易 (需要Tick数据)
df = detect_block_trades(df, volume_multiplier=5, price_deviation=0.02, window=20)
```

##### 7. 综合成交量特征计算

系统提供了综合计算函数，可以一次性计算所有成交量相关特征：

```python
from processing.feature_engine.volume_features import calculate_volume_features

# 计算所有成交量特征
df = calculate_volume_features(
    df, 
    float_shares=100000000,  # 流通股本
    vwap_windows=[15, 30, 60, 240],  # VWAP窗口
    volume_ma_windows=[5, 20, 50, 200],  # 成交量移动平均窗口
    volume_change_windows=[5, 10, 20]  # 成交量变化率窗口
)
```

##### 成交量特征使用建议

**基础组合**
- 趋势确认: `Volume_MA_20`, `Volume_Change_Rate_5`, `AD_Line`
- 超买超卖: `MFI`, `Volume_Spike`
- 流动性分析: `Turnover_Rate`, `Volume_Ratio_20`

**高级组合**
- 量价背离: 结合价格趋势和成交量背离指标
- 资金流向: `AD_Line`, `MFI`, `VWAP` 综合判断
- 异常检测: 大单检测 + 异常交易识别

**策略应用**
- 突破策略: 价格突破 + 成交量确认
- 反转策略: 量价背离 + 技术指标确认
- 流动性策略: 基于换手率和成交量活跃度

#### 技术指标特征

1. **震荡指标**
   - RSI (相对强弱指标)
   - MACD (移动平均收敛发散)
   - 随机指标 (Stochastic)
   - 威廉指标 (Williams %R)

2. **技术形态**
   - 锤子线/上吊线
   - 十字星
   - 吞没形态

3. **季节性特征**
   - 时间特征 (星期、月份、季度)
   - 月度季节性
   - 周内季节性

#### 跨资产特征

1. **价差和比率**
   - 资产间价差
   - 价格比率
   - 相关性指标

### 内存优化

- 自动数据类型优化
- 分块处理大数据集
- 垃圾回收管理
- 内存使用监控

### 数据输出

- 支持多种格式: Feather, Parquet, CSV
- 自动生成数据报告和特征报告
- 详细的处理日志

## 特征使用示例

### Python API使用

```python
from processing.feature_engine.price_features import calculate_price_features
from processing.feature_engine.volume_features import calculate_volume_features

# 计算价格特征
df_price = calculate_price_features(df)

# 计算成交量特征
df_volume = calculate_volume_features(df, float_shares=100000000)

# 合并特征
df_features = pd.concat([df_price, df_volume], axis=1)
```

### 成交量特征测试

运行测试脚本验证成交量特征功能：

```bash
# 基础测试 (不依赖TA-Lib)
python scripts/simple_volume_test.py

# 完整测试 (需要TA-Lib)
python scripts/test_volume_features.py

# 使用示例
python examples/volume_features_example.py
```

### 特征选择建议

#### 基础特征组合
- **趋势跟踪**: SMA_20, SMA_50, EMA_20, BB_Position
- **动量策略**: Return_5d, Return_20d, RSI_14, MACD
- **波动率策略**: HistVol_20d, ATR_20d, BB_Width
- **成交量策略**: Volume_MA_20, MFI, Volume_Change_Rate_5, AD_Line

#### 高级特征组合
- **多因子模型**: 结合价格、成交量、技术指标
- **机器学习**: 使用所有特征进行特征选择
- **深度学习**: 时序特征 + 技术指标

## 配置说明

### 数据库配置

在 `config/settings.py` 中可以配置：

- 数据库连接参数
- SSH隧道设置
- 数据处理参数
- 内存优化设置

### 日志配置

日志配置在 `config/logging.conf` 中，支持：

- 控制台输出
- 文件输出
- 日志轮转
- 不同级别的日志记录

## 性能优化建议

### 低配服务器优化

1. **内存管理**
   - 使用较小的chunk_size (如10000)
   - 启用内存优化功能
   - 定期进行垃圾回收

2. **查询优化**
   - 合理设置查询时间范围
   - 使用测试模式验证
   - 避免同时处理过多数据

3. **存储优化**
   - 使用Feather或Parquet格式
   - 启用压缩
   - 定期清理临时文件

4. **特征计算优化**
   - 按需计算特征类型
   - 使用向量化操作
   - 避免重复计算

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查网络连接
   - 验证数据库配置
   - 确认用户权限

2. **内存不足**
   - 减小chunk_size
   - 使用测试模式
   - 检查系统内存使用

3. **数据处理缓慢**
   - 优化查询条件
   - 使用SSH隧道
   - 调整处理参数

4. **特征计算错误**
   - 检查数据质量
   - 验证特征参数
   - 查看错误日志

5. **TA-Lib安装问题**
   - Windows: 下载预编译包
   - Linux: 安装系统依赖后编译
   - macOS: 使用Homebrew安装

## 开发指南

### 添加新功能

1. 在相应的模块中添加功能
2. 更新配置文件
3. 添加测试用例
4. 更新文档

### 代码规范

- 使用类型注解
- 添加详细的文档字符串
- 遵循PEP 8规范
- 进行代码审查

### 特征开发

1. **添加新特征**
   - 在相应的特征模块中添加新方法
   - 更新特征字典文档
   - 添加单元测试

2. **特征优化**
   - 使用向量化操作
   - 优化计算效率
   - 减少内存使用

3. **特征验证**
   - 检查特征合理性
   - 验证计算正确性
   - 测试边界条件

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至项目维护者 