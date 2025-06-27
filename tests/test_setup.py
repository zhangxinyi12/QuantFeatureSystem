#!/usr/bin/env python3
"""
项目设置测试脚本
验证项目结构和基本功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from config.settings import DATABASE_CONFIG, DATA_CONFIG
        print("✓ 配置文件导入成功")
    except ImportError as e:
        print(f"✗ 配置文件导入失败: {e}")
        return False
    
    try:
        from src.database.connector import JuyuanDB
        print("✓ 数据库连接器导入成功")
    except ImportError as e:
        print(f"✗ 数据库连接器导入失败: {e}")
        return False
    
    try:
        from src.database.queries import StockQueries
        print("✓ 查询模块导入成功")
    except ImportError as e:
        print(f"✗ 查询模块导入失败: {e}")
        return False
    
    try:
        from src.processing.timeseries import TimeSeriesProcessor
        print("✓ 时序处理模块导入成功")
    except ImportError as e:
        print(f"✗ 时序处理模块导入失败: {e}")
        return False
    
    try:
        from src.utils.memory_utils import MemoryManager
        print("✓ 内存工具模块导入成功")
    except ImportError as e:
        print(f"✗ 内存工具模块导入失败: {e}")
        return False
    
    return True

def test_project_structure():
    """测试项目结构"""
    print("\n测试项目结构...")
    
    required_dirs = [
        'config',
        'data_schema', 
        'src/database',
        'src/processing',
        'src/utils',
        'output/reports',
        'output/processed_data',
        'output/logs',
        'scripts'
    ]
    
    required_files = [
        'config/settings.py',
        'config/logging.conf',
        'data_schema/juyuan_dictionary.md',
        'src/database/connector.py',
        'src/database/queries.py',
        'src/processing/timeseries.py',
        'src/utils/memory_utils.py',
        'src/main.py',
        'requirements.txt',
        'README.md',
        '.gitignore'
    ]
    
    all_good = True
    
    # 检查目录
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ 目录存在: {dir_path}")
        else:
            print(f"✗ 目录缺失: {dir_path}")
            all_good = False
    
    # 检查文件
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ 文件存在: {file_path}")
        else:
            print(f"✗ 文件缺失: {file_path}")
            all_good = False
    
    return all_good

def test_configuration():
    """测试配置"""
    print("\n测试配置...")
    
    try:
        from config.settings import DATABASE_CONFIG, DATA_CONFIG, OUTPUT_CONFIG
        
        # 检查数据库配置
        required_db_keys = ['host', 'port', 'user', 'password', 'database']
        for key in required_db_keys:
            if key in DATABASE_CONFIG:
                print(f"✓ 数据库配置: {key}")
            else:
                print(f"✗ 数据库配置缺失: {key}")
                return False
        
        # 检查数据处理配置
        if 'default_start_date' in DATA_CONFIG:
            print("✓ 数据处理配置: default_start_date")
        else:
            print("✗ 数据处理配置缺失: default_start_date")
            return False
        
        # 检查输出配置
        for key, path in OUTPUT_CONFIG.items():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
                print(f"✓ 输出目录创建: {key}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        from src.database.queries import StockQueries
        
        # 测试SQL生成
        sql = StockQueries.get_stock_quote_history(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        if 'SELECT' in sql and 'QT_DailyQuote' in sql:
            print("✓ SQL查询生成成功")
        else:
            print("✗ SQL查询生成失败")
            return False
        
        # 测试内存管理器
        from src.utils.memory_utils import MemoryManager
        memory_manager = MemoryManager()
        memory_info = memory_manager.get_memory_info()
        
        if 'total' in memory_info and 'used' in memory_info:
            print("✓ 内存管理器工作正常")
        else:
            print("✗ 内存管理器异常")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("项目设置测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("项目结构", test_project_structure),
        ("配置检查", test_configuration),
        ("基本功能", test_basic_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目设置完成。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查项目设置。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 