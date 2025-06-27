# 聚源数据库字段说明

## 主要数据表

### 1. QT_DailyQuote (日线行情表)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| InnerCode | int | 内部编码 |
| TradingDay | date | 交易日期 |
| PrevClosePrice | decimal | 前收盘价 |
| OpenPrice | decimal | 开盘价 |
| HighPrice | decimal | 最高价 |
| LowPrice | decimal | 最低价 |
| ClosePrice | decimal | 收盘价 |
| TurnoverVolume | bigint | 成交量 |
| TurnoverValue | decimal | 成交金额 |
| VWAP | decimal | 成交量加权平均价 |
| AdjFactor | decimal | 复权因子 |

### 2. SecuMain (证券主表)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| InnerCode | int | 内部编码 |
| SecuCode | varchar | 证券代码 |
| SecuMarket | int | 交易市场 |
| SecuCategory | int | 证券类别 |
| ListedState | int | 上市状态 |
| ListedDate | date | 上市日期 |
| DelistedDate | date | 退市日期 |

### 3. QT_PriceLimit (涨跌停价格表)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| InnerCode | int | 内部编码 |
| TradingDay | date | 交易日期 |
| PriceCeiling | decimal | 涨停价 |
| PriceFloor | decimal | 跌停价 |

### 4. QT_StockPerformance (股票表现表)
| 字段名 | 类型 | 说明 |
|--------|------|------|
| InnerCode | int | 内部编码 |
| TradingDay | date | 交易日期 |
| Ifsuspend | int | 是否停牌 |

## 市场代码说明
- 83: 上海证券交易所
- 90: 深圳证券交易所

## 证券类别说明
- 1: 股票
- 2: 基金
- 3: 债券

## 上市状态说明
- 1: 正常上市
- 2: 暂停上市
- 3: 终止上市

## 复权类型说明
- 前复权: 以当前价格为基准向前复权
- 后复权: 以历史价格为基准向后复权
- 不复权: 原始价格数据 