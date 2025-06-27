"""
数据库连接器
基于聚源数据库的连接管理，支持SSH隧道连接
"""

import pymysql
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import logging
from typing import Optional, Dict, Any
import time

from config.settings import DATABASE_CONFIG, SSH_TUNNEL_CONFIG, SSH_DATABASE_CONFIG
from src.utils.ssh_tunnel import SSHTunnelManager, test_ssh_connection, test_tunnel_connection

logger = logging.getLogger('database')

class JuyuanDB:
    """聚源数据库连接器"""
    
    def __init__(self, use_ssh_tunnel: bool = False):
        """
        初始化数据库连接
        
        Args:
            use_ssh_tunnel: 是否使用SSH隧道连接
        """
        self.use_ssh_tunnel = use_ssh_tunnel
        self.conn = None
        self.engine = None
        self.ssh_tunnel_manager = None
        
        if use_ssh_tunnel:
            self.config = SSH_DATABASE_CONFIG.copy()
            self.ssh_tunnel_manager = SSHTunnelManager(SSH_TUNNEL_CONFIG)
        else:
            self.config = DATABASE_CONFIG.copy()
        
        self._setup_connection()
    
    def _setup_connection(self):
        """设置数据库连接"""
        try:
            if self.use_ssh_tunnel:
                self._setup_ssh_tunnel()
            
            # 创建PyMySQL连接
            self.conn = pymysql.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                charset=self.config['charset'],
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True,
                connect_timeout=30,
                read_timeout=60
            )
            
            # 创建SQLAlchemy引擎
            encoded_password = quote_plus(self.config['password'])
            connection_string = (
                f"mysql+pymysql://{self.config['user']}:{encoded_password}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            
            self.engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                echo=False
            )
            
            logger.info(f"数据库连接建立成功 {'(通过SSH隧道)' if self.use_ssh_tunnel else ''}")
            
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            if self.use_ssh_tunnel and self.ssh_tunnel_manager:
                self.ssh_tunnel_manager.close_tunnel()
            raise
    
    def _setup_ssh_tunnel(self):
        """设置SSH隧道"""
        if not self.ssh_tunnel_manager:
            raise RuntimeError("SSH隧道管理器未初始化")
        
        try:
            # 测试SSH连接
            if not test_ssh_connection(SSH_TUNNEL_CONFIG):
                raise Exception("SSH连接测试失败")
            
            # 创建隧道
            if not self.ssh_tunnel_manager.create_tunnel():
                raise Exception("SSH隧道创建失败")
            
            # 等待隧道建立
            time.sleep(2)
            
            # 测试隧道连接
            if not self.ssh_tunnel_manager.is_tunnel_active():
                raise Exception("SSH隧道连接测试失败")
            
            logger.info("SSH隧道建立成功")
            
        except Exception as e:
            logger.error(f"SSH隧道设置失败: {str(e)}")
            raise
    
    def execute(self, sql: str) -> bool:
        """
        执行SQL语句（非查询操作）
        
        Args:
            sql: SQL语句
            
        Returns:
            bool: 执行是否成功
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                self.conn.commit()
                logger.debug(f"SQL执行成功: {sql[:100]}...")
                return True
        except Exception as e:
            logger.error(f"SQL执行失败: {str(e)}")
            return False
    
    def query(self, sql: str) -> list:
        """
        执行SQL查询并返回结果
        
        Args:
            sql: SQL查询语句
            
        Returns:
            list: 查询结果列表
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
                logger.debug(f"查询成功，返回 {len(result)} 条记录")
                return result
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            return []
    
    def read_sql(self, sql: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        使用pandas读取SQL查询结果
        
        Args:
            sql: SQL查询语句
            chunk_size: 分块大小，用于大数据集
            
        Returns:
            pd.DataFrame: 查询结果DataFrame
        """
        try:
            if chunk_size:
                # 分块读取大数据集
                chunks = []
                for chunk in pd.read_sql(sql, self.engine, chunksize=chunk_size):
                    chunks.append(chunk)
                result = pd.concat(chunks, ignore_index=True)
            else:
                result = pd.read_sql(sql, self.engine)
            
            logger.info(f"数据读取成功，共 {len(result)} 行，{len(result.columns)} 列")
            return result
            
        except Exception as e:
            logger.error(f"数据读取失败: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"连接测试失败: {str(e)}")
            return False
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """获取表信息"""
        try:
            sql = f"DESCRIBE {table_name}"
            result = self.query(sql)
            return {
                'table_name': table_name,
                'columns': result
            }
        except Exception as e:
            logger.error(f"获取表信息失败: {str(e)}")
            return {}
    
    def get_table_count(self, table_name: str) -> int:
        """获取表记录数"""
        try:
            sql = f"SELECT COUNT(*) as count FROM {table_name}"
            result = self.query(sql)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"获取表记录数失败: {str(e)}")
            return 0
    
    def close(self):
        """关闭数据库连接"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("数据库连接已关闭")
            
            if self.engine:
                self.engine.dispose()
                logger.info("数据库引擎已关闭")
            
            if self.use_ssh_tunnel and self.ssh_tunnel_manager:
                self.ssh_tunnel_manager.close_tunnel()
                
        except Exception as e:
            logger.error(f"关闭连接时出错: {str(e)}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

def test_database_connection(use_ssh_tunnel: bool = False) -> bool:
    """
    测试数据库连接
    
    Args:
        use_ssh_tunnel: 是否使用SSH隧道
        
    Returns:
        bool: 连接是否成功
    """
    try:
        with JuyuanDB(use_ssh_tunnel=use_ssh_tunnel) as db:
            if db.test_connection():
                logger.info("数据库连接测试成功")
                return True
            else:
                logger.error("数据库连接测试失败")
                return False
    except Exception as e:
        logger.error(f"数据库连接测试异常: {str(e)}")
        return False

def get_database_info(use_ssh_tunnel: bool = False) -> Dict[str, Any]:
    """
    获取数据库信息
    
    Args:
        use_ssh_tunnel: 是否使用SSH隧道
        
    Returns:
        Dict: 数据库信息
    """
    try:
        with JuyuanDB(use_ssh_tunnel=use_ssh_tunnel) as db:
            # 获取数据库版本
            version_sql = "SELECT VERSION() as version"
            version_result = db.query(version_sql)
            
            # 获取当前数据库
            database_sql = "SELECT DATABASE() as database_name"
            database_result = db.query(database_sql)
            
            # 获取表列表
            tables_sql = "SHOW TABLES"
            tables_result = db.query(tables_sql)
            
            return {
                'version': version_result[0]['version'] if version_result else 'Unknown',
                'database': database_result[0]['database_name'] if database_result else 'Unknown',
                'tables_count': len(tables_result),
                'connection_type': 'SSH Tunnel' if use_ssh_tunnel else 'Direct'
            }
    except Exception as e:
        logger.error(f"获取数据库信息失败: {str(e)}")
        return {}

if __name__ == "__main__":
    # 测试代码
    print("=== 测试直接连接 ===")
    if test_database_connection(use_ssh_tunnel=False):
        info = get_database_info(use_ssh_tunnel=False)
        print(f"数据库信息: {info}")
    
    print("\n=== 测试SSH隧道连接 ===")
    if test_database_connection(use_ssh_tunnel=True):
        info = get_database_info(use_ssh_tunnel=True)
        print(f"数据库信息: {info}") 