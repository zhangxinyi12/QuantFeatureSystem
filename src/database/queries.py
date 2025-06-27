"""
SQL查询语句模块
包含各种股票数据查询的SQL语句
"""

from typing import List, Optional
from config.settings import DATA_CONFIG

class StockQueries:
    """股票数据查询语句"""
    
    @staticmethod
    def get_stock_quote_history(
        start_date: str,
        end_date: str,
        market_codes: Optional[List[int]] = None,
        include_suspended: bool = False
    ) -> str:
        """
        获取股票历史行情数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            market_codes: 市场代码列表
            include_suspended: 是否包含停牌股票
            
        Returns:
            str: SQL查询语句
        """
        if market_codes is None:
            market_codes = DATA_CONFIG['market_codes']
        
        market_condition = f"c.SecuMarket in ({','.join(map(str, market_codes))})"
        
        # 基础字段
        quote_fields = [
            'a.TradingDay', 'a.PrevClosePrice', 'a.OpenPrice', 
            'a.HighPrice', 'a.LowPrice', 'a.ClosePrice', 
            'a.TurnoverVolume', 'a.TurnoverValue', 'a.VWAP', 'a.AdjFactor'
        ]
        
        # 涨跌停价格字段
        limit_fields = ['b.PriceCeiling', 'b.PriceFloor']
        
        # 股票信息字段
        stock_fields = ['c.SecuCode', 'c.SecuMarket', 'd.Ifsuspend']
        
        all_fields = ', '.join(stock_fields + quote_fields + limit_fields)
        
        sql = f"""
        SELECT {all_fields}
        FROM QT_DailyQuote a
        LEFT JOIN QT_PriceLimit b
            ON a.InnerCode = b.InnerCode AND a.TradingDay = b.TradingDay
        LEFT JOIN QT_StockPerformance d
            ON a.InnerCode = d.InnerCode AND a.TradingDay = d.TradingDay
        LEFT JOIN SecuMain c
            ON a.InnerCode = c.InnerCode
        WHERE c.SecuCategory = 1 
            AND {market_condition}
            AND c.ListedState = 1
            AND a.TradingDay BETWEEN '{start_date}' AND '{end_date}'
        """
        
        if not include_suspended:
            sql += " AND (d.Ifsuspend IS NULL OR d.Ifsuspend = 0)"
        
        sql += " ORDER BY c.SecuCode, a.TradingDay"
        
        return sql
    
    @staticmethod
    def get_stock_list(market_codes: Optional[List[int]] = None) -> str:
        """
        获取股票列表
        
        Args:
            market_codes: 市场代码列表
            
        Returns:
            str: SQL查询语句
        """
        if market_codes is None:
            market_codes = DATA_CONFIG['market_codes']
        
        market_condition = f"SecuMarket in ({','.join(map(str, market_codes))})"
        
        sql = f"""
        SELECT InnerCode, SecuCode, SecuMarket, ListedDate, DelistedDate
        FROM SecuMain
        WHERE SecuCategory = 1 
            AND {market_condition}
            AND ListedState = 1
        ORDER BY SecuCode
        """
        
        return sql
    
    @staticmethod
    def get_trading_dates(start_date: str, end_date: str) -> str:
        """
        获取交易日期列表
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            str: SQL查询语句
        """
        sql = f"""
        SELECT DISTINCT TradingDay
        FROM QT_DailyQuote
        WHERE TradingDay BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY TradingDay
        """
        
        return sql
    
    @staticmethod
    def get_stock_adj_factor(
        secu_codes: List[str],
        start_date: str,
        end_date: str
    ) -> str:
        """
        获取股票复权因子
        
        Args:
            secu_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            str: SQL查询语句
        """
        secu_codes_str = "','".join(secu_codes)
        
        sql = f"""
        SELECT c.SecuCode, a.TradingDay, a.AdjFactor
        FROM QT_DailyQuote a
        LEFT JOIN SecuMain c ON a.InnerCode = c.InnerCode
        WHERE c.SecuCode IN ('{secu_codes_str}')
            AND a.TradingDay BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY c.SecuCode, a.TradingDay
        """
        
        return sql
    
    @staticmethod
    def get_stock_quote_by_codes(
        secu_codes: List[str],
        start_date: str,
        end_date: str
    ) -> str:
        """
        根据股票代码获取行情数据
        
        Args:
            secu_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            str: SQL查询语句
        """
        secu_codes_str = "','".join(secu_codes)
        
        quote_fields = [
            'a.TradingDay', 'a.PrevClosePrice', 'a.OpenPrice', 
            'a.HighPrice', 'a.LowPrice', 'a.ClosePrice', 
            'a.TurnoverVolume', 'a.TurnoverValue', 'a.VWAP', 'a.AdjFactor'
        ]
        
        limit_fields = ['b.PriceCeiling', 'b.PriceFloor']
        stock_fields = ['c.SecuCode', 'c.SecuMarket', 'd.Ifsuspend']
        
        all_fields = ', '.join(stock_fields + quote_fields + limit_fields)
        
        sql = f"""
        SELECT {all_fields}
        FROM QT_DailyQuote a
        LEFT JOIN QT_PriceLimit b
            ON a.InnerCode = b.InnerCode AND a.TradingDay = b.TradingDay
        LEFT JOIN QT_StockPerformance d
            ON a.InnerCode = d.InnerCode AND a.TradingDay = d.TradingDay
        LEFT JOIN SecuMain c
            ON a.InnerCode = c.InnerCode
        WHERE c.SecuCode IN ('{secu_codes_str}')
            AND a.TradingDay BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY c.SecuCode, a.TradingDay
        """
        
        return sql
    
    @staticmethod
    def get_market_summary(date: str) -> str:
        """
        获取市场汇总数据
        
        Args:
            date: 交易日期
            
        Returns:
            str: SQL查询语句
        """
        sql = f"""
        SELECT 
            c.SecuMarket,
            COUNT(DISTINCT c.SecuCode) as stock_count,
            SUM(a.TurnoverVolume) as total_volume,
            SUM(a.TurnoverValue) as total_value,
            AVG(a.ClosePrice) as avg_price
        FROM QT_DailyQuote a
        LEFT JOIN SecuMain c ON a.InnerCode = c.InnerCode
        WHERE c.SecuCategory = 1 
            AND c.SecuMarket IN (83, 90)
            AND c.ListedState = 1
            AND a.TradingDay = '{date}'
        GROUP BY c.SecuMarket
        """
        
        return sql 