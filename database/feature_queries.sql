-- =====================================================
-- 量化特征系统 - 特征提取SQL集合
-- 涵盖行情、财务、行业、债券、融资融券等维度
-- 适用于量化投研团队的特征工程需求
-- =====================================================

-- =====================================================
-- 一、行情技术指标（日频）
-- =====================================================

-- 1.1 基础行情 + 技术指标 + 资金流向
SELECT 
    a.TradingDate AS 日期,
    a.InnerCode AS 股票代码,
    a.ClosePrice AS 收盘价,
    a.OpenPrice AS 开盘价,
    a.HighPrice AS 最高价,
    a.LowPrice AS 最低价,
    a.TurnoverVolume AS 成交量,
    a.TurnoverValue AS 成交额,
    a.ChangePCT AS 涨跌幅,
    a.Amplitude AS 振幅,
    -- 技术指标
    b.MACD AS MACD指标,
    b.MACD_Signal AS MACD信号线,
    b.MACD_Histogram AS MACD柱状图,
    c.RSI AS RSI相对强弱,
    c.KDJ_K AS KDJ_K值,
    c.KDJ_D AS KDJ_D值,
    c.KDJ_J AS KDJ_J值,
    c.BB_Upper AS 布林带上轨,
    c.BB_Middle AS 布林带中轨,
    c.BB_Lower AS 布林带下轨,
    c.BB_Width AS 布林带宽度,
    c.BB_Position AS 布林带位置,
    -- 资金流向
    d.NetInflow AS 主力净流入_万元,
    d.MainNetInflow AS 主力净流入_万元,
    d.RetailNetInflow AS 散户净流入_万元,
    d.ForeignNetInflow AS 外资净流入_万元
FROM 
    `境内股票复权行情表` a
LEFT JOIN `境内股票强弱与趋向技术指标` b 
    ON a.InnerCode = b.InnerCode AND a.TradingDate = b.TradingDate
LEFT JOIN `境内股票摆动与反趋向技术指标` c 
    ON a.InnerCode = c.InnerCode AND a.TradingDate = c.TradingDate
LEFT JOIN `股票交易资金流向` d 
    ON a.InnerCode = d.InnerCode AND a.TradingDate = d.TradingDate
WHERE 
    a.TradingDate BETWEEN '2020-01-01' AND '2023-12-31'
    AND a.ClosePrice > 0
    AND a.TurnoverVolume > 0;

-- 1.2 波动率特征计算
SELECT 
    InnerCode,
    TradingDate,
    ClosePrice,
    -- 历史波动率（20日）
    STDDEV(ClosePrice) OVER (
        PARTITION BY InnerCode 
        ORDER BY TradingDate 
        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
    ) AS Volatility_20D,
    -- 历史波动率（60日）
    STDDEV(ClosePrice) OVER (
        PARTITION BY InnerCode 
        ORDER BY TradingDate 
        ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
    ) AS Volatility_60D,
    -- 对数收益率波动率（20日）
    STDDEV(LN(ClosePrice / LAG(ClosePrice, 1) OVER (PARTITION BY InnerCode ORDER BY TradingDate))) OVER (
        PARTITION BY InnerCode 
        ORDER BY TradingDate 
        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
    ) AS LogReturn_Volatility_20D,
    -- 真实波幅（ATR）
    AVG(GREATEST(HighPrice - LowPrice, 
                 ABS(HighPrice - LAG(ClosePrice, 1) OVER (PARTITION BY InnerCode ORDER BY TradingDate)),
                 ABS(LowPrice - LAG(ClosePrice, 1) OVER (PARTITION BY InnerCode ORDER BY TradingDate)))) OVER (
        PARTITION BY InnerCode 
        ORDER BY TradingDate 
        ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
    ) AS ATR_14D
FROM `境内股票复权行情表`
WHERE TradingDate BETWEEN '2020-01-01' AND '2023-12-31';

-- 1.3 移动平均与趋势特征
SELECT 
    InnerCode,
    TradingDate,
    ClosePrice,
    -- 移动平均
    AVG(ClosePrice) OVER (
        PARTITION BY InnerCode 
        ORDER BY TradingDate 
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS MA_5D,
    AVG(ClosePrice) OVER (
        PARTITION BY InnerCode 
        ORDER BY TradingDate 
        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
    ) AS MA_20D,
    AVG(ClosePrice) OVER (
        PARTITION BY InnerCode 
        ORDER BY TradingDate 
        ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
    ) AS MA_60D,
    -- 价格位置
    (ClosePrice - MIN(LowPrice) OVER (
        PARTITION BY InnerCode 
        ORDER BY TradingDate 
        ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
    )) / (MAX(HighPrice) OVER (
        PARTITION BY InnerCode 
        ORDER BY TradingDate 
        ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
    ) - MIN(LowPrice) OVER (
        PARTITION BY InnerCode 
        ORDER BY TradingDate 
        ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
    )) AS Price_Position_60D,
    -- 动量指标
    ClosePrice / LAG(ClosePrice, 5) OVER (PARTITION BY InnerCode ORDER BY TradingDate) - 1 AS Momentum_5D,
    ClosePrice / LAG(ClosePrice, 20) OVER (PARTITION BY InnerCode ORDER BY TradingDate) - 1 AS Momentum_20D
FROM `境内股票复权行情表`
WHERE TradingDate BETWEEN '2020-01-01' AND '2023-12-31';

-- =====================================================
-- 二、财务基本面指标（季频）
-- =====================================================

-- 2.1 基础财务指标
SELECT 
    f.InfoPublDate AS 财报发布日期,
    f.EndDate AS 报告期,
    f.InnerCode AS 股票代码,
    -- 盈利能力
    f.BasicEPS AS 每股收益,
    f.DilutedEPS AS 稀释每股收益,
    f.ROE AS 净资产收益率,
    f.ROA AS 总资产收益率,
    f.GrossProfitMargin AS 毛利率,
    f.NetProfitMargin AS 净利率,
    -- 成长性
    f.RevenueGrowthRate AS 营收增长率,
    f.NetProfitGrowthRate AS 净利润增长率,
    -- 估值指标
    f.PE AS 市盈率,
    f.PB AS 市净率,
    f.PS AS 市销率,
    f.PCF AS 市现率,
    -- 财务质量
    f.DebtToEquityRatio AS 资产负债率,
    f.CurrentRatio AS 流动比率,
    f.QuickRatio AS 速动比率,
    f.AssetTurnover AS 总资产周转率,
    -- 行业信息
    d.IndustryName AS 行业名称,
    d.IndustryCode AS 行业代码,
    i.IndustryPE AS 行业市盈率,
    i.IndustryPB AS 行业市净率,
    -- 业绩预告
    p.ForecastType AS 业绩预告类型,
    p.ForecastNetProfit AS 预告净利润,
    p.ForecastGrowthRate AS 预告增长率
FROM 
    `公司最新财务指标` f
JOIN `公司行业划分表` d 
    ON f.InnerCode = d.InnerCode
LEFT JOIN `行业市盈率_中证发布` i 
    ON d.IndustryCode = i.IndustryCode 
    AND f.InfoPublDate = i.TradingDate
LEFT JOIN `业绩预告` p 
    ON f.InnerCode = p.InnerCode 
    AND f.EndDate = p.EndDate
WHERE 
    f.EndDate = '2023-06-30'
    AND f.InfoPublDate IS NOT NULL;

-- 2.2 财务指标衍生特征
SELECT 
    f1.InnerCode,
    f1.InfoPublDate,
    f1.EndDate,
    -- 财务指标变化
    f1.ROE - f2.ROE AS ROE_Change,
    f1.GrossProfitMargin - f2.GrossProfitMargin AS GrossMargin_Change,
    f1.DebtToEquityRatio - f2.DebtToEquityRatio AS DebtRatio_Change,
    -- 财务指标排名（行业内）
    RANK() OVER (PARTITION BY f1.IndustryCode ORDER BY f1.ROE DESC) AS ROE_Rank,
    RANK() OVER (PARTITION BY f1.IndustryCode ORDER BY f1.GrossProfitMargin DESC) AS GrossMargin_Rank,
    -- 财务指标分位数
    PERCENT_RANK() OVER (PARTITION BY f1.IndustryCode ORDER BY f1.ROE) AS ROE_Percentile,
    PERCENT_RANK() OVER (PARTITION BY f1.IndustryCode ORDER BY f1.PE) AS PE_Percentile,
    -- 相对估值
    f1.PE / i.IndustryPE AS PE_Relative,
    f1.PB / i.IndustryPB AS PB_Relative
FROM 
    `公司最新财务指标` f1
JOIN `公司行业划分表` d ON f1.InnerCode = d.InnerCode
LEFT JOIN `公司最新财务指标` f2 
    ON f1.InnerCode = f2.InnerCode 
    AND f2.EndDate = DATE_SUB(f1.EndDate, INTERVAL 1 YEAR)
LEFT JOIN `行业市盈率_中证发布` i 
    ON d.IndustryCode = i.IndustryCode 
    AND f1.InfoPublDate = i.TradingDate
WHERE 
    f1.EndDate = '2023-06-30'
    AND f2.EndDate IS NOT NULL;

-- =====================================================
-- 三、债券及衍生品指标
-- =====================================================

-- 3.1 可转债特征
SELECT
    b.TradingDate AS 日期,
    b.BondCode AS 转债代码,
    s.StockName AS 正股名称,
    s.InnerCode AS 正股代码,
    b.ClosePrice AS 转债收盘价,
    b.ConvertPrice AS 转股价,
    b.PremiumRate AS 转股溢价率,
    b.YTM AS 到期收益率,
    b.Duration AS 久期,
    b.Convexity AS 凸性,
    -- 正股信息
    q.ClosePrice AS 正股收盘价,
    q.TurnoverVolume AS 正股成交量,
    q.TurnoverValue AS 正股成交额,
    -- 转债规模
    c.BondBalance AS 债券余额,
    c.ConvertedShares AS 已转股数,
    c.RemainingShares AS 剩余转股数
FROM 
    `可转换债券行情` b
JOIN `可转债转股及规模变动情况` c 
    ON b.BondCode = c.BondCode 
    AND b.TradingDate = c.TradingDate
JOIN `证券主表` s 
    ON c.StockInnerCode = s.InnerCode
LEFT JOIN `股票日行情表现` q 
    ON s.InnerCode = q.InnerCode
    AND b.TradingDate = q.TradingDate
WHERE 
    b.TradingDate = '2023-12-31'
    AND b.ClosePrice > 0;

-- 3.2 转债衍生特征
SELECT 
    b.BondCode,
    b.TradingDate,
    b.ClosePrice,
    b.ConvertPrice,
    b.PremiumRate,
    -- 转债隐含波动率（简化计算）
    SQRT(2 * PI()) * ABS(b.PremiumRate) / 100 AS Implied_Volatility,
    -- 转债相对强度
    b.ClosePrice / AVG(b.ClosePrice) OVER (
        PARTITION BY b.BondCode 
        ORDER BY b.TradingDate 
        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
    ) AS Bond_Relative_Strength,
    -- 转股价值
    (q.ClosePrice / b.ConvertPrice) * 100 AS Conversion_Value,
    -- 转债折溢价
    (b.ClosePrice - (q.ClosePrice / b.ConvertPrice) * 100) / (q.ClosePrice / b.ConvertPrice) * 100 AS Bond_Discount_Premium
FROM 
    `可转换债券行情` b
JOIN `可转债转股及规模变动情况` c 
    ON b.BondCode = c.BondCode 
    AND b.TradingDate = c.TradingDate
JOIN `证券主表` s 
    ON c.StockInnerCode = s.InnerCode
LEFT JOIN `股票日行情表现` q 
    ON s.InnerCode = q.InnerCode
    AND b.TradingDate = q.TradingDate
WHERE 
    b.TradingDate BETWEEN '2023-01-01' AND '2023-12-31';

-- =====================================================
-- 四、市场情绪指标
-- =====================================================

-- 4.1 融资融券特征
SELECT
    m.TradingDate AS 日期,
    m.InnerCode AS 股票代码,
    m.MarginBalance AS 融资余额_万元,
    m.ShortSellingBalance AS 融券余额_万元,
    m.MarginBuyAmount AS 融资买入额_万元,
    m.ShortSellingAmount AS 融券卖出额_万元,
    m.MarginRepayAmount AS 融资偿还额_万元,
    m.ShortSellingRepayAmount AS 融券偿还额_万元,
    -- 融资融券比率
    CASE 
        WHEN m.ShortSellingBalance > 0 
        THEN m.MarginBalance / m.ShortSellingBalance 
        ELSE NULL 
    END AS Margin_Short_Ratio,
    -- 融资余额变化
    m.MarginBalance - LAG(m.MarginBalance, 1) OVER (PARTITION BY m.InnerCode ORDER BY m.TradingDate) AS Margin_Balance_Change,
    -- 融券余额变化
    m.ShortSellingBalance - LAG(m.ShortSellingBalance, 1) OVER (PARTITION BY m.InnerCode ORDER BY m.TradingDate) AS Short_Balance_Change,
    -- 大宗交易
    t.TotalValue AS 大宗交易额_万元,
    t.TotalVolume AS 大宗交易量,
    t.AvgPremium AS 大宗交易溢价率
FROM 
    `融资融券交易明细` m
LEFT JOIN (
    SELECT 
        InnerCode, 
        TradingDate, 
        SUM(TradeValue) AS TotalValue,
        SUM(TradeVolume) AS TotalVolume,
        AVG(PremiumRate) AS AvgPremium
    FROM `大宗交易成交明细`
    GROUP BY InnerCode, TradingDate
) t ON m.InnerCode = t.InnerCode AND m.TradingDate = t.TradingDate
WHERE 
    m.TradingDate >= '2023-01-01';

-- 4.2 市场情绪衍生特征
SELECT 
    m.InnerCode,
    m.TradingDate,
    m.MarginBalance,
    m.ShortSellingBalance,
    -- 融资融券余额趋势
    m.MarginBalance / AVG(m.MarginBalance) OVER (
        PARTITION BY m.InnerCode 
        ORDER BY m.TradingDate 
        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
    ) AS Margin_Balance_Trend,
    -- 融资融券余额分位数
    PERCENT_RANK() OVER (
        PARTITION BY m.InnerCode 
        ORDER BY m.MarginBalance
    ) AS Margin_Balance_Percentile,
    -- 融资融券活跃度
    (m.MarginBuyAmount + m.ShortSellingAmount) / (m.MarginBalance + m.ShortSellingBalance) AS Margin_Activity_Ratio
FROM 
    `融资融券交易明细` m
WHERE 
    m.TradingDate >= '2023-01-01'
    AND m.MarginBalance > 0;

-- =====================================================
-- 五、行业相对强度特征
-- =====================================================

-- 5.1 个股相对行业超额收益
SELECT 
    s.TradingDate,
    s.InnerCode,
    s.ClosePrice,
    s.ChangePCT AS Stock_Return,
    i.IndexClose,
    i.ChangePCT AS Index_Return,
    -- 相对超额收益
    s.ChangePCT - i.ChangePCT AS Alpha_1D,
    -- 5日相对超额收益
    (s.ClosePrice / LAG(s.ClosePrice, 5) OVER w) - 
    (i.IndexClose / LAG(i.IndexClose, 5) OVER w) AS Alpha_5D,
    -- 20日相对超额收益
    (s.ClosePrice / LAG(s.ClosePrice, 20) OVER w) - 
    (i.IndexClose / LAG(i.IndexClose, 20) OVER w) AS Alpha_20D,
    -- 相对强度
    s.ClosePrice / LAG(s.ClosePrice, 20) OVER w / 
    (i.IndexClose / LAG(i.IndexClose, 20) OVER w) AS Relative_Strength_20D
FROM `股票日行情表现` s
JOIN `公司行业划分表` c ON s.InnerCode = c.InnerCode
JOIN `指数行情` i ON c.IndustryIndexCode = i.IndexCode 
    AND s.TradingDate = i.TradingDate
WINDOW w AS (PARTITION BY s.InnerCode ORDER BY s.TradingDate)
WHERE s.TradingDate BETWEEN '2023-01-01' AND '2023-12-31';

-- 5.2 行业轮动特征
SELECT 
    i.TradingDate,
    i.IndexCode,
    i.IndexName,
    i.IndexClose,
    i.ChangePCT,
    -- 行业动量
    i.IndexClose / LAG(i.IndexClose, 20) OVER (PARTITION BY i.IndexCode ORDER BY i.TradingDate) - 1 AS Industry_Momentum_20D,
    -- 行业相对强度
    i.IndexClose / LAG(i.IndexClose, 60) OVER (PARTITION BY i.IndexCode ORDER BY i.TradingDate) / 
    (SELECT AVG(IndexClose / LAG(IndexClose, 60) OVER (PARTITION BY IndexCode ORDER BY TradingDate)) 
     FROM `指数行情` 
     WHERE TradingDate = i.TradingDate) AS Industry_Relative_Strength,
    -- 行业排名
    RANK() OVER (PARTITION BY i.TradingDate ORDER BY i.ChangePCT DESC) AS Industry_Rank_Daily,
    RANK() OVER (PARTITION BY i.TradingDate ORDER BY 
        i.IndexClose / LAG(i.IndexClose, 20) OVER (PARTITION BY i.IndexCode ORDER BY i.TradingDate) DESC) AS Industry_Rank_Momentum
FROM `指数行情` i
WHERE i.TradingDate BETWEEN '2023-01-01' AND '2023-12-31';

-- =====================================================
-- 六、模型训练数据整合（宽表）
-- =====================================================

-- 6.1 日频特征宽表
CREATE TABLE feature_daily_wide AS
SELECT 
    -- 基础信息
    q.TradingDate,
    q.InnerCode,
    q.StockName,
    q.IndustryCode,
    q.IndustryName,
    
    -- 行情特征
    q.ClosePrice,
    q.TurnoverVolume,
    q.TurnoverValue,
    q.ChangePCT,
    q.Amplitude,
    
    -- 技术指标
    q.MACD,
    q.RSI,
    q.KDJ_K,
    q.BB_Width,
    q.BB_Position,
    
    -- 资金流向
    q.NetInflow,
    q.MainNetInflow,
    
    -- 波动率特征
    v.Volatility_20D,
    v.Volatility_60D,
    v.ATR_14D,
    
    -- 趋势特征
    t.MA_5D,
    t.MA_20D,
    t.MA_60D,
    t.Price_Position_60D,
    t.Momentum_5D,
    t.Momentum_20D,
    
    -- 财务特征（最新）
    f.ROE,
    f.GrossProfitMargin,
    f.PE,
    f.PB,
    f.ROE_Rank,
    f.PE_Relative,
    
    -- 转债特征
    cb.PremiumRate,
    cb.Implied_Volatility,
    
    -- 市场情绪
    m.MarginBalance,
    m.ShortSellingBalance,
    m.Margin_Short_Ratio,
    m.Margin_Balance_Trend,
    
    -- 行业特征
    ind.Alpha_5D,
    ind.Relative_Strength_20D,
    ind.Industry_Momentum_20D,
    
    -- 未来收益（目标变量）
    LEAD(q.ChangePCT, 1) OVER (PARTITION BY q.InnerCode ORDER BY q.TradingDate) AS Future_Return_1D,
    LEAD(q.ChangePCT, 5) OVER (PARTITION BY q.InnerCode ORDER BY q.TradingDate) AS Future_Return_5D,
    LEAD(q.ChangePCT, 20) OVER (PARTITION BY q.InnerCode ORDER BY q.TradingDate) AS Future_Return_20D

FROM (
    -- 基础行情查询
    SELECT 
        a.TradingDate,
        a.InnerCode,
        s.StockName,
        c.IndustryCode,
        c.IndustryName,
        a.ClosePrice,
        a.TurnoverVolume,
        a.TurnoverValue,
        a.ChangePCT,
        a.Amplitude,
        b.MACD,
        b.RSI,
        b.KDJ_K,
        b.BB_Width,
        b.BB_Position,
        d.NetInflow,
        d.MainNetInflow
    FROM `境内股票复权行情表` a
    JOIN `证券主表` s ON a.InnerCode = s.InnerCode
    JOIN `公司行业划分表` c ON a.InnerCode = c.InnerCode
    LEFT JOIN `境内股票强弱与趋向技术指标` b 
        ON a.InnerCode = b.InnerCode AND a.TradingDate = b.TradingDate
    LEFT JOIN `股票交易资金流向` d 
        ON a.InnerCode = d.InnerCode AND a.TradingDate = d.TradingDate
    WHERE a.TradingDate BETWEEN '2020-01-01' AND '2023-12-31'
) q

-- 波动率特征
LEFT JOIN (
    SELECT 
        InnerCode,
        TradingDate,
        STDDEV(ClosePrice) OVER (
            PARTITION BY InnerCode 
            ORDER BY TradingDate 
            ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
        ) AS Volatility_20D,
        STDDEV(ClosePrice) OVER (
            PARTITION BY InnerCode 
            ORDER BY TradingDate 
            ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
        ) AS Volatility_60D,
        AVG(GREATEST(HighPrice - LowPrice, 
                     ABS(HighPrice - LAG(ClosePrice, 1) OVER (PARTITION BY InnerCode ORDER BY TradingDate)),
                     ABS(LowPrice - LAG(ClosePrice, 1) OVER (PARTITION BY InnerCode ORDER BY TradingDate)))) OVER (
            PARTITION BY InnerCode 
            ORDER BY TradingDate 
            ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
        ) AS ATR_14D
    FROM `境内股票复权行情表`
    WHERE TradingDate BETWEEN '2020-01-01' AND '2023-12-31'
) v ON q.InnerCode = v.InnerCode AND q.TradingDate = v.TradingDate

-- 趋势特征
LEFT JOIN (
    SELECT 
        InnerCode,
        TradingDate,
        AVG(ClosePrice) OVER (
            PARTITION BY InnerCode 
            ORDER BY TradingDate 
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS MA_5D,
        AVG(ClosePrice) OVER (
            PARTITION BY InnerCode 
            ORDER BY TradingDate 
            ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
        ) AS MA_20D,
        AVG(ClosePrice) OVER (
            PARTITION BY InnerCode 
            ORDER BY TradingDate 
            ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
        ) AS MA_60D,
        (ClosePrice - MIN(LowPrice) OVER (
            PARTITION BY InnerCode 
            ORDER BY TradingDate 
            ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
        )) / (MAX(HighPrice) OVER (
            PARTITION BY InnerCode 
            ORDER BY TradingDate 
            ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
        ) - MIN(LowPrice) OVER (
            PARTITION BY InnerCode 
            ORDER BY TradingDate 
            ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
        )) AS Price_Position_60D,
        ClosePrice / LAG(ClosePrice, 5) OVER (PARTITION BY InnerCode ORDER BY TradingDate) - 1 AS Momentum_5D,
        ClosePrice / LAG(ClosePrice, 20) OVER (PARTITION BY InnerCode ORDER BY TradingDate) - 1 AS Momentum_20D
    FROM `境内股票复权行情表`
    WHERE TradingDate BETWEEN '2020-01-01' AND '2023-12-31'
) t ON q.InnerCode = t.InnerCode AND q.TradingDate = t.TradingDate

-- 财务特征
LEFT JOIN (
    SELECT 
        f1.InnerCode,
        f1.InfoPublDate,
        f1.ROE,
        f1.GrossProfitMargin,
        f1.PE,
        f1.PB,
        RANK() OVER (PARTITION BY f1.IndustryCode ORDER BY f1.ROE DESC) AS ROE_Rank,
        f1.PE / i.IndustryPE AS PE_Relative
    FROM `公司最新财务指标` f1
    JOIN `公司行业划分表` d ON f1.InnerCode = d.InnerCode
    LEFT JOIN `行业市盈率_中证发布` i 
        ON d.IndustryCode = i.IndustryCode 
        AND f1.InfoPublDate = i.TradingDate
    WHERE f1.EndDate = '2023-06-30'
) f ON q.InnerCode = f.InnerCode AND q.TradingDate >= f.InfoPublDate

-- 转债特征
LEFT JOIN (
    SELECT 
        c.StockInnerCode AS InnerCode,
        b.TradingDate,
        b.PremiumRate,
        SQRT(2 * PI()) * ABS(b.PremiumRate) / 100 AS Implied_Volatility
    FROM `可转换债券行情` b
    JOIN `可转债转股及规模变动情况` c 
        ON b.BondCode = c.BondCode 
        AND b.TradingDate = c.TradingDate
    WHERE b.TradingDate BETWEEN '2020-01-01' AND '2023-12-31'
) cb ON q.InnerCode = cb.InnerCode AND q.TradingDate = cb.TradingDate

-- 市场情绪特征
LEFT JOIN (
    SELECT 
        InnerCode,
        TradingDate,
        MarginBalance,
        ShortSellingBalance,
        CASE 
            WHEN ShortSellingBalance > 0 
            THEN MarginBalance / ShortSellingBalance 
            ELSE NULL 
        END AS Margin_Short_Ratio,
        MarginBalance / AVG(MarginBalance) OVER (
            PARTITION BY InnerCode 
            ORDER BY TradingDate 
            ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
        ) AS Margin_Balance_Trend
    FROM `融资融券交易明细`
    WHERE TradingDate BETWEEN '2020-01-01' AND '2023-12-31'
) m ON q.InnerCode = m.InnerCode AND q.TradingDate = m.TradingDate

-- 行业特征
LEFT JOIN (
    SELECT 
        s.InnerCode,
        s.TradingDate,
        (s.ChangePCT - i.ChangePCT) AS Alpha_1D,
        (s.ClosePrice / LAG(s.ClosePrice, 5) OVER w) - 
        (i.IndexClose / LAG(i.IndexClose, 5) OVER w) AS Alpha_5D,
        s.ClosePrice / LAG(s.ClosePrice, 20) OVER w / 
        (i.IndexClose / LAG(i.IndexClose, 20) OVER w) AS Relative_Strength_20D,
        i.IndexClose / LAG(i.IndexClose, 20) OVER (PARTITION BY i.IndexCode ORDER BY i.TradingDate) - 1 AS Industry_Momentum_20D
    FROM `股票日行情表现` s
    JOIN `公司行业划分表` c ON s.InnerCode = c.InnerCode
    JOIN `指数行情` i ON c.IndustryIndexCode = i.IndexCode 
        AND s.TradingDate = i.TradingDate
    WINDOW w AS (PARTITION BY s.InnerCode ORDER BY s.TradingDate)
    WHERE s.TradingDate BETWEEN '2020-01-01' AND '2023-12-31'
) ind ON q.InnerCode = ind.InnerCode AND q.TradingDate = ind.TradingDate;

-- 6.2 创建索引优化查询性能
CREATE INDEX idx_feature_daily_date ON feature_daily_wide(TradingDate);
CREATE INDEX idx_feature_daily_stock ON feature_daily_wide(InnerCode, TradingDate);
CREATE INDEX idx_feature_daily_industry ON feature_daily_wide(IndustryCode, TradingDate);

-- =====================================================
-- 七、特征筛选与评估SQL
-- =====================================================

-- 7.1 计算IC值（信息系数）
SELECT 
    feature_name,
    AVG(ic_value) AS IC_Mean,
    STDDEV(ic_value) AS IC_Std,
    AVG(ic_value) / STDDEV(ic_value) AS ICIR,
    COUNT(*) AS Sample_Count,
    AVG(CASE WHEN ABS(ic_value) > 0.05 THEN 1 ELSE 0 END) AS Significant_Ratio
FROM (
    SELECT 
        TradingDate,
        'ROE' AS feature_name,
        CORR(ROE, Future_Return_5D) AS ic_value
    FROM feature_daily_wide 
    WHERE ROE IS NOT NULL AND Future_Return_5D IS NOT NULL
    
    UNION ALL
    
    SELECT 
        TradingDate,
        'RSI' AS feature_name,
        CORR(RSI, Future_Return_5D) AS ic_value
    FROM feature_daily_wide 
    WHERE RSI IS NOT NULL AND Future_Return_5D IS NOT NULL
    
    UNION ALL
    
    SELECT 
        TradingDate,
        'Volatility_20D' AS feature_name,
        CORR(Volatility_20D, Future_Return_5D) AS ic_value
    FROM feature_daily_wide 
    WHERE Volatility_20D IS NOT NULL AND Future_Return_5D IS NOT NULL
    
    UNION ALL
    
    SELECT 
        TradingDate,
        'Margin_Short_Ratio' AS feature_name,
        CORR(Margin_Short_Ratio, Future_Return_5D) AS ic_value
    FROM feature_daily_wide 
    WHERE Margin_Short_Ratio IS NOT NULL AND Future_Return_5D IS NOT NULL
) ic_calc
GROUP BY feature_name
ORDER BY ABS(IC_Mean) DESC;

-- 7.2 特征稳定性分析
SELECT 
    feature_name,
    period,
    AVG(ic_value) AS IC_Mean,
    STDDEV(ic_value) AS IC_Std,
    STDDEV(ic_value) / ABS(AVG(ic_value)) AS IC_Stability
FROM (
    SELECT 
        CASE 
            WHEN TradingDate < '2021-01-01' THEN '2020'
            WHEN TradingDate < '2022-01-01' THEN '2021'
            WHEN TradingDate < '2023-01-01' THEN '2022'
            ELSE '2023'
        END AS period,
        'ROE' AS feature_name,
        CORR(ROE, Future_Return_5D) AS ic_value
    FROM feature_daily_wide 
    WHERE ROE IS NOT NULL AND Future_Return_5D IS NOT NULL
    GROUP BY period
    
    UNION ALL
    
    SELECT 
        CASE 
            WHEN TradingDate < '2021-01-01' THEN '2020'
            WHEN TradingDate < '2022-01-01' THEN '2021'
            WHEN TradingDate < '2023-01-01' THEN '2022'
            ELSE '2023'
        END AS period,
        'RSI' AS feature_name,
        CORR(RSI, Future_Return_5D) AS ic_value
    FROM feature_daily_wide 
    WHERE RSI IS NOT NULL AND Future_Return_5D IS NOT NULL
    GROUP BY period
) stability_calc
GROUP BY feature_name, period
ORDER BY feature_name, period;

-- 7.3 特征重要性排序（基于ICIR）
WITH feature_importance AS (
    SELECT 
        feature_name,
        AVG(ic_value) AS IC_Mean,
        STDDEV(ic_value) AS IC_Std,
        AVG(ic_value) / STDDEV(ic_value) AS ICIR,
        COUNT(*) AS Sample_Count
    FROM (
        -- 这里可以添加更多特征
        SELECT 
            TradingDate,
            'ROE' AS feature_name,
            CORR(ROE, Future_Return_5D) AS ic_value
        FROM feature_daily_wide 
        WHERE ROE IS NOT NULL AND Future_Return_5D IS NOT NULL
        
        UNION ALL
        
        SELECT 
            TradingDate,
            'RSI' AS feature_name,
            CORR(RSI, Future_Return_5D) AS ic_value
        FROM feature_daily_wide 
        WHERE RSI IS NOT NULL AND Future_Return_5D IS NOT NULL
    ) ic_calc
    GROUP BY feature_name
)
SELECT 
    feature_name,
    IC_Mean,
    IC_Std,
    ICIR,
    Sample_Count,
    RANK() OVER (ORDER BY ABS(ICIR) DESC) AS Importance_Rank
FROM feature_importance
WHERE ABS(ICIR) > 0.1  -- 筛选有效特征
ORDER BY Importance_Rank;

-- =====================================================
-- 八、性能优化建议
-- =====================================================

/*
1. 分区策略：
   - 按时间分区：PARTITION BY RANGE (YEAR(TradingDate))
   - 按股票分区：PARTITION BY HASH(InnerCode)

2. 索引优化：
   - 主键：(InnerCode, TradingDate)
   - 复合索引：(TradingDate, IndustryCode)
   - 特征索引：对常用特征列建立索引

3. 查询优化：
   - 使用EXPLAIN分析执行计划
   - 避免SELECT *，只查询需要的列
   - 合理使用WHERE条件过滤
   - 使用LIMIT限制结果集大小

4. 数据更新策略：
   - 增量更新：只更新最新数据
   - 批量更新：使用LOAD DATA INFILE
   - 异步更新：使用消息队列

5. 缓存策略：
   - 热点数据缓存：Redis缓存常用查询结果
   - 特征缓存：预计算并缓存特征值
   - 查询缓存：缓存复杂查询结果
*/

-- =====================================================
-- 九、监控与维护SQL
-- =====================================================

-- 9.1 数据质量检查
SELECT 
    'feature_daily_wide' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT InnerCode) AS unique_stocks,
    COUNT(DISTINCT TradingDate) AS unique_dates,
    MIN(TradingDate) AS min_date,
    MAX(TradingDate) AS max_date,
    AVG(CASE WHEN ClosePrice IS NULL THEN 1 ELSE 0 END) AS null_price_ratio,
    AVG(CASE WHEN ROE IS NULL THEN 1 ELSE 0 END) AS null_roe_ratio
FROM feature_daily_wide;

-- 9.2 特征分布检查
SELECT 
    'ROE' AS feature_name,
    COUNT(*) AS total_count,
    AVG(ROE) AS mean_value,
    STDDEV(ROE) AS std_value,
    MIN(ROE) AS min_value,
    MAX(ROE) AS max_value,
    COUNT(CASE WHEN ROE > 3 * STDDEV(ROE) + AVG(ROE) THEN 1 END) AS outlier_count
FROM feature_daily_wide
WHERE ROE IS NOT NULL

UNION ALL

SELECT 
    'RSI' AS feature_name,
    COUNT(*) AS total_count,
    AVG(RSI) AS mean_value,
    STDDEV(RSI) AS std_value,
    MIN(RSI) AS min_value,
    MAX(RSI) AS max_value,
    COUNT(CASE WHEN RSI > 100 OR RSI < 0 THEN 1 END) AS outlier_count
FROM feature_daily_wide
WHERE RSI IS NOT NULL;

-- 9.3 特征相关性检查
SELECT 
    CORR(ROE, RSI) AS ROE_RSI_Corr,
    CORR(ROE, Volatility_20D) AS ROE_Vol_Corr,
    CORR(RSI, Volatility_20D) AS RSI_Vol_Corr,
    CORR(PE, PB) AS PE_PB_Corr
FROM feature_daily_wide
WHERE ROE IS NOT NULL 
    AND RSI IS NOT NULL 
    AND Volatility_20D IS NOT NULL 
    AND PE IS NOT NULL 
    AND PB IS NOT NULL; 