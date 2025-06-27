# 量化特征字典

本文档详细说明了量化特征系统中所有特征的计算方法和含义。

## 特征分类

### 1. 价格衍生特征 (Price Features)

#### 1.1 收益率特征 (Returns)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `Return_{N}d` | `(Close_t - Close_{t-N}) / Close_{t-N}` | N日绝对收益率 |
| `LogReturn_{N}d` | `ln(Close_t / Close_{t-N})` | N日对数收益率 |
| `CumReturn_{N}d` | `(Close_t / Close_{t-N}) - 1` | N日累积收益率 |

**常用周期**: 1, 5, 20, 60, 252天

#### 1.2 动量特征 (Momentum)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `Momentum_{N}d` | `Close_t - Close_{t-N}` | N日价格动量 |
| `ROC_{N}d` | `(Close_t / Close_{t-N} - 1) * 100` | N日变化率 |
| `RelMomentum_{N}d` | `(Close_t - SMA_N) / SMA_N` | 相对动量 |

**常用周期**: 5, 10, 20, 60天

#### 1.3 波动率特征 (Volatility)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `HistVol_{N}d` | `std(log_returns) * sqrt(252)` | 历史波动率 |
| `RealizedVol_{N}d` | `sqrt(sum(log_returns^2) * 252/N)` | 已实现波动率 |
| `ParkinsonVol_{N}d` | `sqrt(sum(ln(H/L)^2) / (4N*ln(2)) * 252)` | Parkinson波动率 |
| `GKVol_{N}d` | `sqrt(0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2)` | Garman-Klass波动率 |
| `ATR_{N}d` | `mean(TR)` | 平均真实波幅 |

**常用窗口**: 20, 60, 252天

#### 1.4 趋势指标 (Trend Indicators)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `SMA_{N}` | `mean(Close, N)` | N日简单移动平均 |
| `EMA_{N}` | `EMA(Close, N)` | N日指数移动平均 |
| `PricePos_{N}` | `(Close - SMA_N) / SMA_N` | 价格相对位置 |
| `BB_Upper` | `SMA_20 + 2*std_20` | 布林带上轨 |
| `BB_Lower` | `SMA_20 - 2*std_20` | 布林带下轨 |
| `BB_Width` | `(BB_Upper - BB_Lower) / SMA_20` | 布林带宽度 |
| `BB_Position` | `(Close - BB_Lower) / (BB_Upper - BB_Lower)` | 布林带位置 |

**常用窗口**: 20, 50, 200天

#### 1.5 支撑阻力位 (Support & Resistance)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `Support_{N}d` | `min(Low, N)` | N日支撑位 |
| `Resistance_{N}d` | `max(High, N)` | N日阻力位 |
| `PricePosition_{N}d` | `(Close - Support) / (Resistance - Support)` | 价格位置 |
| `Fib_{0.236}` | `Swing_High - 0.236 * Range` | 斐波那契回撤位 |
| `Fib_{0.382}` | `Swing_High - 0.382 * Range` | 斐波那契回撤位 |
| `Fib_{0.5}` | `Swing_High - 0.5 * Range` | 斐波那契回撤位 |
| `Fib_{0.618}` | `Swing_High - 0.618 * Range` | 斐波那契回撤位 |
| `Fib_{0.786}` | `Swing_High - 0.786 * Range` | 斐波那契回撤位 |

### 2. 成交量衍生特征 (Volume Features)

#### 2.1 成交量指标 (Volume Indicators)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `Volume_MA_{N}` | `mean(Volume, N)` | N日成交量移动平均 |
| `Volume_Ratio_{N}` | `Volume / Volume_MA_N` | 成交量比率 |
| `Volume_Std_{N}` | `std(Volume, N)` | 成交量标准差 |
| `Volume_Change_{N}` | `(Volume_t - Volume_{t-N}) / Volume_{t-N}` | 成交量变化率 |
| `VWAP` | `sum(Price * Volume) / sum(Volume)` | 成交量加权平均价格 |

**常用窗口**: 5, 20, 60天

#### 2.2 价量关系特征 (Volume-Price Features)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `PriceVolume_Corr_{N}` | `corr(Price_Change, Volume)` | 价量相关性 |
| `PriceVolume_Divergence_{N}` | `(Price_Rel - Volume_Rel)` | 价量背离 |
| `VWAP_Momentum_{N}` | `(Close - VWAP) / VWAP` | VWAP动量 |
| `MFI` | `100 - 100/(1 + Positive_MF/Negative_MF)` | 资金流量指标 |

#### 2.3 成交量模式 (Volume Patterns)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `Volume_ZScore` | `(Volume - Volume_MA) / Volume_Std` | 成交量Z分数 |
| `Volume_Spike` | `Volume_ZScore > 2` | 成交量异常 |
| `Volume_Trend` | `trend(Volume, 5)` | 成交量趋势 |
| `Volume_Percentile` | `rank(Volume, 252)` | 成交量百分位 |

### 3. 技术指标特征 (Technical Indicators)

#### 3.1 震荡指标 (Oscillators)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `RSI_14` | `100 - 100/(1 + RS)` | 相对强弱指标 |
| `MACD` | `EMA_12 - EMA_26` | MACD线 |
| `MACD_Signal` | `EMA(MACD, 9)` | MACD信号线 |
| `MACD_Histogram` | `MACD - MACD_Signal` | MACD柱状图 |
| `Stoch_K` | `100 * (Close - Low_14) / (High_14 - Low_14)` | 随机指标K值 |
| `Stoch_D` | `SMA(Stoch_K, 3)` | 随机指标D值 |
| `Williams_R` | `-100 * (High_14 - Close) / (High_14 - Low_14)` | 威廉指标 |

#### 3.2 技术形态 (Patterns)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `Hammer` | 下影线 > 2*实体, 上影线 < 实体, 阳线 | 锤子线 |
| `Hanging_Man` | 下影线 > 2*实体, 上影线 < 实体, 阴线 | 上吊线 |
| `Doji` | 实体 < 0.1*总长度 | 十字星 |
| `Bullish_Engulfing` | 阳线完全吞没前一根阴线 | 看涨吞没 |
| `Bearish_Engulfing` | 阴线完全吞没前一根阳线 | 看跌吞没 |

#### 3.3 季节性特征 (Seasonality)
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `DayOfWeek` | `0-6` | 星期几 |
| `Month` | `1-12` | 月份 |
| `Quarter` | `1-4` | 季度 |
| `DayOfYear` | `1-366` | 年内第几天 |
| `Monthly_Seasonality` | `mean(Returns by Month)` | 月度季节性 |
| `Weekly_Seasonality` | `mean(Returns by DayOfWeek)` | 周内季节性 |

### 4. 跨资产特征 (Cross-Asset Features)

#### 4.1 价差和比率
| 特征名 | 计算方法 | 说明 |
|--------|----------|------|
| `Spread_vs_{Asset}` | `Price_A - Price_B` | 价差 |
| `Ratio_vs_{Asset}` | `Price_A / Price_B` | 价格比率 |
| `Corr_vs_{Asset}` | `corr(Price_A, Price_B, 20)` | 相关性 |

## 特征使用建议

### 1. 特征选择策略

#### 基础特征组合
- **趋势跟踪**: SMA_20, SMA_50, EMA_20, BB_Position
- **动量策略**: Return_5d, Return_20d, RSI_14, MACD
- **波动率策略**: HistVol_20d, ATR_20d, BB_Width
- **成交量策略**: Volume_Ratio_20, MFI, Volume_Spike

#### 高级特征组合
- **多因子模型**: 结合价格、成交量、技术指标
- **机器学习**: 使用所有特征进行特征选择
- **深度学习**: 时序特征 + 技术指标

### 2. 特征工程最佳实践

#### 数据预处理
1. **缺失值处理**: 前向填充或插值
2. **异常值处理**: 3σ法则或IQR方法
3. **标准化**: Z-score或Min-Max标准化
4. **去趋势化**: 差分或移动平均去趋势

#### 特征选择
1. **相关性分析**: 去除高度相关特征
2. **重要性排序**: 基于模型特征重要性
3. **稳定性检验**: 不同时间段特征稳定性
4. **过拟合检测**: 交叉验证避免过拟合

#### 特征组合
1. **交互特征**: 价格特征 × 成交量特征
2. **比率特征**: 不同周期特征比率
3. **滞后特征**: 历史特征滞后值
4. **滚动特征**: 滚动窗口统计量

### 3. 性能优化

#### 计算优化
- 使用向量化操作
- 并行计算大量特征
- 缓存中间结果
- 使用Numba加速

#### 内存优化
- 分块处理大数据集
- 使用适当的数据类型
- 及时释放内存
- 监控内存使用

### 4. 特征监控

#### 质量监控
- 特征覆盖率
- 特征稳定性
- 特征相关性
- 特征重要性

#### 性能监控
- 计算时间
- 内存使用
- 特征更新频率
- 系统负载

## 示例代码

```python
from processing.feature_engine.price_features import calculate_price_features
from processing.feature_engine.volume_features import calculate_volume_features
from processing.feature_engine.technical_indicators import calculate_technical_features

# 计算价格特征
df_price = calculate_price_features(df)

# 计算成交量特征
df_volume = calculate_volume_features(df, float_shares=100000000)

# 计算技术指标
df_technical = calculate_technical_features(df)

# 合并所有特征
df_features = pd.concat([df_price, df_volume, df_technical], axis=1)
```

## 注意事项

1. **数据质量**: 确保输入数据质量，处理缺失值和异常值
2. **计算效率**: 大数据集时考虑分块处理
3. **特征解释**: 理解每个特征的经济含义
4. **过拟合风险**: 避免使用过多特征导致过拟合
5. **实时性**: 考虑特征计算的实时性要求 