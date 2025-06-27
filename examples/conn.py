import pymysql
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

class jydb:
    
    def __init__(self):
        # 数据库连接参数
        self.host = '172.18.122.83'
        self.port = 3306
        # 修改后的连接参数
        # self.host = '127.0.0.1'  # 通过 SSH 隧道本地转发
        # self.port = 3307         # 与 SSH 隧道的本地端口一致
        self.user = 'jydb_ro'
        self.password = '1qaz@WSX'
        self.db = 'gildata'
        
        # 创建数据库连接
        self.conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.db,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        # 创建 SQLAlchemy 引擎（用于 pandas.read_sql）
        # 对密码中的特殊字符进行URL编码
        encoded_password = quote_plus(self.password)
        self.engine = create_engine(
            f'mysql+pymysql://{self.user}:{encoded_password}@{self.host}:{self.port}/{self.db}'
        )
    
    def execute(self, sql):
        """执行 SQL 查询（用于非查询操作，如 INSERT、UPDATE）"""
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            self.conn.commit()
    
    def query(self, sql):
        """执行 SQL 查询并返回结果（用于 SELECT）"""
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()
    
    def read_sql(self, sql):
        """使用 pandas 读取 SQL 查询结果"""
        return pd.read_sql(sql, self.engine)
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()