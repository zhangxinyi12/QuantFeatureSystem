#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征流水线命令行入口脚本
提供简单的命令行接口来运行特征流水线
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.feature_pipeline import FeaturePipeline

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征构造流水线')
    parser.add_argument('--input', type=str, required=True, help='输入数据文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径（JSON格式）')
    parser.add_argument('--standardize', action='store_true', help='是否标准化特征')
    parser.add_argument('--remove-low-variance', action='store_true', default=True, 
                       help='是否移除低方差特征')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config:
        import json
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
    config.update({
        'standardize_features': args.standardize,
        'remove_low_variance': args.remove_low_variance
    })
    
    # 创建流水线
    pipeline = FeaturePipeline(config)
    
    try:
        # 运行流水线
        result_df = pipeline.run_pipeline(args.input, args.output, config)
        print(f"流水线运行成功，输出数据形状: {result_df.shape}")
        
    except Exception as e:
        print(f"流水线运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 