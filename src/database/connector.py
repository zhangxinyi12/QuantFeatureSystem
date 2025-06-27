"""
数据库连接器
基于聚源数据库的连接管理
"""

import pymysql
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import logging
from typing import Optional, Dict, Any
import time

from config.settings import DATABASE_CONFIG, SSH_TUNNEL_CONFIG

logger = logging.getLogger('database')

class JuyuanDB:
    """聚源数据库连接器"""
    
    def __init__(self, use_ssh_tunnel: bool = False):
        """
        初始化数据库连接
        
        Args:
            use_ssh_tunnel: 是否使用SSH隧道连接
        """
        self.config = DATABASE_CONFIG.copy()
        self.use_ssh_tunnel = use_ssh_tunnel
        self.conn = None
        self.engine = None
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
                autocommit=True
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
                pool_recycle=3600
            )
            
            logger.info("数据库连接建立成功")
            
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise
    
    def _setup_ssh_tunnel(self):
        """设置SSH隧道（如果需要）"""
        if not SSH_TUNNEL_CONFIG['enabled']:
            return
            
        try:
            import paramiko
            import socket
            
            # 创建SSH隧道
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            ssh.connect(
                SSH_TUNNEL_CONFIG['ssh_host'],
                port=SSH_TUNNEL_CONFIG['ssh_port'],
                username=SSH_TUNNEL_CONFIG['ssh_user']
            )
            
            # 创建本地端口转发
            ssh.get_transport().request_port_forward(
                '',  # 本地地址
                SSH_TUNNEL_CONFIG['local_port'],
                SSH_TUNNEL_CONFIG['remote_host'],
                SSH_TUNNEL_CONFIG['remote_port']
            )
            
            # 更新连接配置
            self.config['host'] = '127.0.0.1'
            self.config['port'] = SSH_TUNNEL_CONFIG['local_port']
            
            logger.info("SSH隧道建立成功")
            
        except ImportError:
            logger.warning("paramiko未安装，无法使用SSH隧道")
        except Exception as e:
            logger.error(f"SSH隧道建立失败: {str(e)}")
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
    
    def close(self):
        """关闭数据库连接"""
        try:
            if self.conn:
                self.conn.close()
            if self.engine:
                self.engine.dispose()
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭连接时出错: {str(e)}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close() 