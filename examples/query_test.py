xiimport conn

def query_stock_quote_hist(start='2015-01-01', end='2024-10-16'):
    conn_jy = conn.jydb()
    
    q_key = ['TradingDay', 'PrevClosePrice', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'TurnoverVolume', 'TurnoverValue']
    l_key = ['PriceCeiling', 'PriceFloor']
    q_key = ','.join(['a.'+s for s in q_key])
    l_key = ','.join(['b.'+s for s in l_key])
    
    data_sql = f"""select c.SecuCode,c.SecuMarket,d.Ifsuspend,{q_key},{l_key} from QT_DailyQuote a
                left join QT_PriceLimit b
                on a.InnerCode=b.InnerCode and a.TradingDay=b.TradingDay
                left join QT_StockPerformance d
                on a.InnerCode=d.InnerCode and a.TradingDay=d.TradingDay
                left join SecuMain c
                on a.InnerCode=c.InnerCode
                where (c.SecuCategory=1 and c.SecuMarket in (83,90) and c.ListedState=1)
                and a.TradingDay between '{start}' and '{end}'
             """
    
    daily_info = conn_jy.read_sql(data_sql)
    daily_info.to_feather('stock_quote_hist.ftr')
    conn_jy.close()
    return daily_info

# 执行查询
df = query_stock_quote_hist(start='2023-01-01', end='2023-12-31')
print(f"数据已保存，共 {len(df)} 行")