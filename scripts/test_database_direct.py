#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库直接连接测试脚本
在服务器上直接连接数据库，不使用SSH隧道
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connector import test_database_connection, get_database_info, JuyuanDB
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    print("=== 数据库直接连接测试 ===")
    
    # 测试直接数据库连接
    print("\n1. 测试直接数据库连接...")
    if test_database_connection(use_ssh_tunnel=False):
        print("✅ 直接数据库连接成功")
        
        # 获取数据库信息
        print("\n2. 获取数据库信息...")
        info = get_database_info(use_ssh_tunnel=False)
        if info:
            print(f"✅ 数据库信息获取成功")
            print(f"   数据库: {info.get('database', 'N/A')}")
            print(f"   用户: {info.get('user', 'N/A')}")
            print(f"   主机: {info.get('host', 'N/A')}")
            print(f"   端口: {info.get('port', 'N/A')}")
        else:
            print("❌ 数据库信息获取失败")
    else:
        print("❌ 直接数据库连接失败")
        return False
    
    # 测试简单查询
    print("\n3. 测试简单查询...")
    try:
        with JuyuanDB(use_ssh_tunnel=False) as db:
            result = db.query("SELECT 1 as test")
            if result:
                print("✅ 简单查询成功")
                print(f"   查询结果: {result}")
            else:
                print("❌ 简单查询失败")
    except Exception as e:
        print(f"❌ 查询测试异常: {e}")
        return False
    
    print("\n✅ 数据库访问测试成功")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 