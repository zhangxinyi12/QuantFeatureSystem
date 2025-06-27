import conn
import pandas as pd
import sys

def query_stock_quote_hist(start='2015-01-01', end='2024-10-16'):
    """
    查询股票历史行情数据
    
    Args:
        start (str): 开始日期，格式：'YYYY-MM-DD'
        end (str): 结束日期，格式：'YYYY-MM-DD'
    
    Returns:
        pandas.DataFrame: 股票历史行情数据
    """
    try:
        print(f"正在连接数据库...")
        conn_jy = conn.jydb()
        print("数据库连接成功")
        
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
        
        print(f"正在查询数据，时间范围：{start} 到 {end}")
        print("SQL查询语句：")
        print(data_sql)
        
        daily_info = conn_jy.read_sql(data_sql)
        
        print(f"查询完成，共获取 {len(daily_info)} 行数据")
        
        if len(daily_info) > 0:
            print("数据预览：")
            print(daily_info.head())
            print(f"数据列：{list(daily_info.columns)}")
        
        # 保存数据
        filename = 'stock_quote_hist.ftr'
        daily_info.to_feather(filename)
        print(f"数据已保存到 {filename}")
        
        conn_jy.close()
        print("数据库连接已关闭")
        
        return daily_info
        
    except Exception as e:
        print(f"查询过程中发生错误：{str(e)}")
        print(f"错误类型：{type(e).__name__}")
        return None

if __name__ == "__main__":
    # 执行查询
    print("开始执行股票历史行情查询...")
    df = query_stock_quote_hist(start='2023-01-01', end='2023-12-31')
    
    if df is not None:
        print(f"查询成功！数据已保存，共 {len(df)} 行")
    else:
        print("查询失败，请检查错误信息")
        sys.exit(1) 