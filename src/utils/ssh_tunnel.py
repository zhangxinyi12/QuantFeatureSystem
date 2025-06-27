#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSH隧道连接工具
用于通过SSH隧道连接远程数据库
"""

import paramiko
import socket
import threading
import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class SSHTunnel:
    """SSH隧道管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化SSH隧道
        
        Args:
            config: SSH隧道配置字典
        """
        self.config = config
        self.ssh_client = None
        self.tunnel_thread = None
        self.tunnel_active = False
        
    def create_tunnel(self) -> bool:
        """
        创建SSH隧道
        
        Returns:
            bool: 隧道创建是否成功
        """
        try:
            # 创建SSH客户端
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 连接SSH服务器
            logger.info(f"正在连接SSH服务器: {self.config['ssh_host']}:{self.config['ssh_port']}")
            self.ssh_client.connect(
                hostname=self.config['ssh_host'],
                port=self.config['ssh_port'],
                username=self.config['ssh_user'],
                password=self.config['ssh_password'],
                timeout=self.config.get('timeout', 30)
            )
            
            # 创建端口转发
            transport = self.ssh_client.get_transport()
            
            # 检查端口是否被占用
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                test_socket.bind(('', int(self.config['local_port'])))
                test_socket.close()
            except OSError as e:
                if e.errno == 48:  # Address already in use
                    logger.warning(f"端口 {self.config['local_port']} 已被占用，尝试清理...")
                    # 尝试关闭可能存在的连接
                    try:
                        cleanup_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        cleanup_socket.settimeout(1)
                        cleanup_socket.connect(('localhost', int(self.config['local_port'])))
                        cleanup_socket.close()
                    except:
                        pass
                    # 等待一下再重试
                    time.sleep(2)
                else:
                    raise e
            
            # 创建本地监听socket
            self.local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.local_socket.bind(('', int(self.config['local_port'])))
            self.local_socket.listen(1)
            
            # 启动隧道线程
            self.tunnel_thread = threading.Thread(target=self._tunnel_worker)
            self.tunnel_thread.daemon = True
            self.tunnel_thread.start()
            
            self.tunnel_active = True
            logger.info(f"SSH隧道创建成功: localhost:{self.config['local_port']} -> "
                       f"{self.config['remote_host']}:{self.config['remote_port']}")
            
            return True
            
        except Exception as e:
            logger.error(f"SSH隧道创建失败: {e}")
            self.close_tunnel()
            return False
    
    def _tunnel_worker(self):
        """隧道工作线程，处理端口转发"""
        try:
            while self.tunnel_active:
                try:
                    # 接受本地连接
                    client_socket, addr = self.local_socket.accept()
                    
                    # 创建SSH通道到远程服务器
                    transport = self.ssh_client.get_transport()
                    channel = transport.open_channel(
                        'direct-tcpip',
                        (self.config['remote_host'], int(self.config['remote_port'])),
                        ('', 0)
                    )
                    
                    # 启动数据转发线程
                    import threading
                    forward_thread = threading.Thread(
                        target=self._forward_data,
                        args=(client_socket, channel)
                    )
                    forward_thread.daemon = True
                    forward_thread.start()
                    
                except Exception as e:
                    if self.tunnel_active:
                        logger.error(f"隧道工作线程错误: {e}")
                    break
        except Exception as e:
            logger.error(f"隧道工作线程异常: {e}")
    
    def _forward_data(self, client_socket, channel):
        """转发数据"""
        try:
            # 双向数据转发
            import threading
            import select
            
            def forward_to_channel():
                try:
                    while True:
                        data = client_socket.recv(4096)
                        if not data:
                            break
                        channel.send(data)
                except:
                    pass
            
            def forward_to_client():
                try:
                    while True:
                        data = channel.recv(4096)
                        if not data:
                            break
                        client_socket.send(data)
                except:
                    pass
            
            # 启动双向转发
            t1 = threading.Thread(target=forward_to_channel)
            t2 = threading.Thread(target=forward_to_client)
            t1.daemon = True
            t2.daemon = True
            t1.start()
            t2.start()
            
            # 等待任一方向结束
            t1.join()
            t2.join()
            
        except Exception as e:
            logger.error(f"数据转发错误: {e}")
        finally:
            try:
                client_socket.close()
                channel.close()
            except:
                pass
    
    def close_tunnel(self):
        """关闭SSH隧道"""
        if hasattr(self, 'local_socket') and self.local_socket:
            try:
                self.local_socket.close()
            except Exception as e:
                logger.error(f"关闭本地监听socket时出错: {e}")
            finally:
                self.local_socket = None
                
        if self.ssh_client:
            try:
                self.ssh_client.close()
                logger.info("SSH隧道已关闭")
            except Exception as e:
                logger.error(f"关闭SSH隧道时出错: {e}")
            finally:
                self.ssh_client = None
                self.tunnel_active = False
    
    def is_tunnel_active(self) -> bool:
        """检查隧道是否活跃"""
        if not self.ssh_client or not self.tunnel_active:
            return False
        
        try:
            # 测试连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', self.config['local_port']))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    @contextmanager
    def tunnel_context(self):
        """
        SSH隧道上下文管理器
        
        Usage:
            with ssh_tunnel.tunnel_context():
                # 在这里使用数据库连接
                pass
        """
        try:
            if self.create_tunnel():
                yield self
            else:
                raise Exception("SSH隧道创建失败")
        finally:
            self.close_tunnel()

class SSHTunnelManager:
    """SSH隧道管理器（单例模式）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not self._initialized and config:
            self.config = config
            self.tunnel = SSHTunnel(config)
            self._initialized = True
    
    def get_tunnel(self) -> SSHTunnel:
        """获取SSH隧道实例"""
        if not hasattr(self, 'tunnel'):
            raise RuntimeError("SSH隧道管理器未初始化，请先传入配置")
        return self.tunnel
    
    def create_tunnel(self) -> bool:
        """创建隧道"""
        return self.get_tunnel().create_tunnel()
    
    def close_tunnel(self):
        """关闭隧道"""
        self.get_tunnel().close_tunnel()
    
    def is_tunnel_active(self) -> bool:
        """检查隧道是否活跃"""
        return self.get_tunnel().is_tunnel_active()

def test_ssh_connection(config: Dict[str, Any]) -> bool:
    """
    测试SSH连接
    
    Args:
        config: SSH配置
        
    Returns:
        bool: 连接是否成功
    """
    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        logger.info(f"测试SSH连接: {config['ssh_host']}:{config['ssh_port']}")
        ssh_client.connect(
            hostname=config['ssh_host'],
            port=config['ssh_port'],
            username=config['ssh_user'],
            password=config['ssh_password'],
            timeout=config.get('timeout', 30)
        )
        
        # 执行简单命令测试连接
        stdin, stdout, stderr = ssh_client.exec_command('echo "SSH连接测试成功"')
        result = stdout.read().decode().strip()
        
        ssh_client.close()
        
        logger.info(f"SSH连接测试成功: {result}")
        return True
        
    except Exception as e:
        logger.error(f"SSH连接测试失败: {e}")
        return False

def test_tunnel_connection(config: Dict[str, Any]) -> bool:
    """
    测试隧道连接
    
    Args:
        config: SSH隧道配置
        
    Returns:
        bool: 隧道连接是否成功
    """
    tunnel = SSHTunnel(config)
    
    try:
        if tunnel.create_tunnel():
            # 测试本地端口是否可访问
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', config['local_port']))
            sock.close()
            
            if result == 0:
                logger.info("隧道连接测试成功")
                return True
            else:
                logger.error("隧道端口不可访问")
                return False
        else:
            return False
    finally:
        tunnel.close_tunnel()

if __name__ == "__main__":
    # 测试代码
    from config.settings import SSH_TUNNEL_CONFIG
    
    # 测试SSH连接
    print("测试SSH连接...")
    if test_ssh_connection(SSH_TUNNEL_CONFIG):
        print("SSH连接测试成功")
    else:
        print("SSH连接测试失败")
    
    # 测试隧道连接
    print("\n测试隧道连接...")
    if test_tunnel_connection(SSH_TUNNEL_CONFIG):
        print("隧道连接测试成功")
    else:
        print("隧道连接测试失败") 