#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSH连接测试脚本
测试服务器连接和数据库访问功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from config.settings import SERVER_CONFIG, SSH_TUNNEL_CONFIG, SSH_DATABASE_CONFIG
from src.utils.ssh_tunnel import test_ssh_connection, test_tunnel_connection, SSHTunnelManager
from src.database.connector import test_database_connection, get_database_info, JuyuanDB

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_server_connection():
    """测试服务器连接"""
    print("=" * 60)
    print("测试服务器连接")
    print("=" * 60)
    
    print(f"服务器信息:")
    print(f"  IP地址: {SERVER_CONFIG['host']}")
    print(f"  用户名: {SERVER_CONFIG['username']}")
    print(f"  端口: {SERVER_CONFIG['port']}")
    print()
    
    # 测试SSH连接
    print("1. 测试SSH连接...")
    if test_ssh_connection(SSH_TUNNEL_CONFIG):
        print("✅ SSH连接测试成功")
    else:
        print("❌ SSH连接测试失败")
        return False
    
    # 测试隧道连接
    print("\n2. 测试SSH隧道连接...")
    if test_tunnel_connection(SSH_TUNNEL_CONFIG):
        print("✅ SSH隧道连接测试成功")
    else:
        print("❌ SSH隧道连接测试失败")
        return False
    
    return True

def test_database_access():
    """测试数据库访问"""
    print("\n" + "=" * 60)
    print("测试数据库访问")
    print("=" * 60)
    
    print("数据库配置信息:")
    print(f"  远程主机: {SSH_TUNNEL_CONFIG['remote_host']}:{SSH_TUNNEL_CONFIG['remote_port']}")
    print(f"  本地端口: localhost:{SSH_TUNNEL_CONFIG['local_port']}")
    print(f"  数据库: {SSH_DATABASE_CONFIG['database']}")
    print(f"  用户: {SSH_DATABASE_CONFIG['user']}")
    print()
    
    # 测试直接连接（如果可能）
    print("1. 测试直接数据库连接...")
    try:
        if test_database_connection(use_ssh_tunnel=False):
            print("✅ 直接数据库连接成功")
            info = get_database_info(use_ssh_tunnel=False)
            print(f"   数据库信息: {info}")
        else:
            print("❌ 直接数据库连接失败（这是正常的，需要通过SSH隧道）")
    except Exception as e:
        print(f"❌ 直接数据库连接异常: {e}")
    
    # 测试SSH隧道数据库连接
    print("\n2. 测试SSH隧道数据库连接...")
    try:
        if test_database_connection(use_ssh_tunnel=True):
            print("✅ SSH隧道数据库连接成功")
            info = get_database_info(use_ssh_tunnel=True)
            print(f"   数据库信息: {info}")
        else:
            print("❌ SSH隧道数据库连接失败")
            return False
    except Exception as e:
        print(f"❌ SSH隧道数据库连接异常: {e}")
        return False
    
    return True

def test_data_query():
    """测试数据查询"""
    print("\n" + "=" * 60)
    print("测试数据查询")
    print("=" * 60)
    
    try:
        with JuyuanDB(use_ssh_tunnel=True) as db:
            # 测试简单查询
            print("1. 测试简单查询...")
            result = db.query("SELECT VERSION() as version")
            if result:
                print(f"✅ 数据库版本: {result[0]['version']}")
            else:
                print("❌ 版本查询失败")
            
            # 测试表列表
            print("\n2. 获取表列表...")
            tables = db.query("SHOW TABLES")
            if tables:
                print(f"✅ 数据库中共有 {len(tables)} 个表")
                print("   前10个表:")
                for i, table in enumerate(tables[:10]):
                    table_name = list(table.values())[0]
                    count = db.get_table_count(table_name)
                    print(f"     {i+1}. {table_name} ({count} 条记录)")
            else:
                print("❌ 获取表列表失败")
            
            # 测试特征查询（使用你的SQL）
            print("\n3. 测试特征查询...")
            feature_sql = """
            SELECT 
                a.TradingDate AS 日期,
                a.InnerCode AS 股票代码,
                a.ClosePrice AS 收盘价,
                a.TurnoverVolume AS 成交量,
                b.MACD AS MACD指标,
                c.RSI AS RSI相对强弱
            FROM 
                `境内股票复权行情表` a
            LEFT JOIN `境内股票强弱与趋向技术指标` b 
                ON a.InnerCode = b.InnerCode AND a.TradingDate = b.TradingDate
            LEFT JOIN `境内股票摆动与反趋向技术指标` c 
                ON a.InnerCode = c.InnerCode AND a.TradingDate = c.TradingDate
            WHERE 
                a.TradingDate = '2023-12-29'
                AND a.ClosePrice > 0
            LIMIT 5
            """
            
            try:
                feature_result = db.query(feature_sql)
                if feature_result:
                    print(f"✅ 特征查询成功，返回 {len(feature_result)} 条记录")
                    print("   示例数据:")
                    for i, row in enumerate(feature_result[:3]):
                        print(f"     {i+1}. {row['股票代码']} - {row['收盘价']} - RSI: {row.get('RSI相对强弱', 'N/A')}")
                else:
                    print("❌ 特征查询失败或无数据")
            except Exception as e:
                print(f"❌ 特征查询异常: {e}")
                
    except Exception as e:
        print(f"❌ 数据查询测试异常: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 开始SSH连接和数据库访问测试")
    print(f"测试时间: {__import__('datetime').datetime.now()}")
    
    # 测试服务器连接
    if not test_server_connection():
        print("\n❌ 服务器连接测试失败，请检查网络和服务器配置")
        return False
    
    # 测试数据库访问
    if not test_database_access():
        print("\n❌ 数据库访问测试失败，请检查数据库配置")
        return False
    
    # 测试数据查询
    if not test_data_query():
        print("\n❌ 数据查询测试失败，请检查SQL语句和表结构")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！SSH隧道和数据库连接正常工作")
    print("=" * 60)
    
    print("\n📋 使用建议:")
    print("1. 在代码中使用 use_ssh_tunnel=True 参数连接数据库")
    print("2. 使用上下文管理器确保连接正确关闭")
    print("3. 定期测试连接状态")
    print("4. 监控SSH隧道稳定性")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生异常: {e}")
        sys.exit(1) 