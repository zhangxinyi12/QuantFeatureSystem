#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源监控工具模块
提供系统资源监控、性能分析、内存管理等功能
"""

import psutil
import time
import threading
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import gc

class SystemMonitor:
    """系统资源监控类"""
    
    def __init__(self, log_interval: int = 60, alert_thresholds: Dict = None):
        """
        初始化系统监控器
        
        参数:
            log_interval: 日志记录间隔（秒）
            alert_thresholds: 告警阈值配置
        """
        self.log_interval = log_interval
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0
        }
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.alert_callbacks = []
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            self.logger.warning("监控已在运行中")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("系统监控已启动")
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("系统监控已停止")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self.get_system_metrics()
                self.metrics_history.append(metrics)
                
                # 检查告警
                self._check_alerts(metrics)
                
                # 记录日志
                self.logger.debug(f"系统指标: {metrics}")
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                self.logger.error(f"监控过程中出现错误: {e}")
                time.sleep(self.log_interval)
                
    def get_system_metrics(self) -> Dict:
        """
        获取系统指标
        
        返回:
            系统指标字典
        """
        timestamp = datetime.now()
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        
        # 网络IO
        network = psutil.net_io_counters()
        
        # 进程信息
        process = psutil.Process()
        
        metrics = {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'network_bytes_sent': network.bytes_sent,
            'network_bytes_recv': network.bytes_recv,
            'process_memory_mb': process.memory_info().rss / (1024**2),
            'process_cpu_percent': process.cpu_percent()
        }
        
        return metrics
        
    def _check_alerts(self, metrics: Dict):
        """检查告警条件"""
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                alert_msg = f"告警: {metric} = {metrics[metric]:.2f} > {threshold}"
                self.logger.warning(alert_msg)
                
                # 调用告警回调函数
                for callback in self.alert_callbacks:
                    try:
                        callback(metric, metrics[metric], threshold)
                    except Exception as e:
                        self.logger.error(f"告警回调函数执行失败: {e}")
                        
    def add_alert_callback(self, callback: Callable):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
        
    def get_metrics_history(self, hours: int = 24) -> pd.DataFrame:
        """
        获取历史指标数据
        
        参数:
            hours: 获取最近多少小时的数据
            
        返回:
            历史指标DataFrame
        """
        if not self.metrics_history:
            return pd.DataFrame()
            
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m['timestamp'] >= cutoff_time]
        
        if not recent_metrics:
            return pd.DataFrame()
            
        df = pd.DataFrame(recent_metrics)
        df.set_index('timestamp', inplace=True)
        return df
        
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """
        获取性能摘要
        
        参数:
            hours: 统计最近多少小时的数据
            
        返回:
            性能摘要字典
        """
        df = self.get_metrics_history(hours)
        if df.empty:
            return {}
            
        summary = {
            'avg_cpu_percent': df['cpu_percent'].mean(),
            'max_cpu_percent': df['cpu_percent'].max(),
            'avg_memory_percent': df['memory_percent'].mean(),
            'max_memory_percent': df['memory_percent'].max(),
            'avg_disk_percent': df['disk_percent'].mean(),
            'max_disk_percent': df['disk_percent'].max(),
            'total_network_sent_gb': df['network_bytes_sent'].sum() / (1024**3),
            'total_network_recv_gb': df['network_bytes_recv'].sum() / (1024**3)
        }
        
        return summary

class MemoryManager:
    """内存管理器"""
    
    def __init__(self):
        """初始化内存管理器"""
        self.logger = logging.getLogger(__name__)
        
    def get_memory_usage(self) -> Dict:
        """
        获取内存使用情况
        
        返回:
            内存使用信息字典
        """
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'process_memory_mb': process.memory_info().rss / (1024**2),
            'process_memory_percent': process.memory_percent()
        }
        
    def optimize_memory(self):
        """优化内存使用"""
        # 强制垃圾回收
        gc.collect()
        
        # 获取优化后的内存使用情况
        before_usage = self.get_memory_usage()
        
        # 执行垃圾回收
        collected = gc.collect()
        
        after_usage = self.get_memory_usage()
        
        freed_mb = before_usage['process_memory_mb'] - after_usage['process_memory_mb']
        
        self.logger.info(f"内存优化完成: 释放了 {freed_mb:.2f} MB, 回收了 {collected} 个对象")
        
        return {
            'freed_mb': freed_mb,
            'collected_objects': collected,
            'before_usage': before_usage,
            'after_usage': after_usage
        }
        
    def check_memory_pressure(self) -> Dict:
        """
        检查内存压力
        
        返回:
            内存压力评估结果
        """
        memory = psutil.virtual_memory()
        
        # 计算内存压力指标
        pressure_level = 'low'
        if memory.percent > 90:
            pressure_level = 'critical'
        elif memory.percent > 80:
            pressure_level = 'high'
        elif memory.percent > 60:
            pressure_level = 'medium'
            
        return {
            'pressure_level': pressure_level,
            'memory_percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'recommendation': self._get_memory_recommendation(pressure_level)
        }
        
    def _get_memory_recommendation(self, pressure_level: str) -> str:
        """获取内存优化建议"""
        recommendations = {
            'low': '内存使用正常，无需优化',
            'medium': '建议监控内存使用趋势',
            'high': '建议进行内存优化或增加内存',
            'critical': '立即进行内存优化，考虑增加系统内存'
        }
        return recommendations.get(pressure_level, '未知状态')

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        """初始化性能分析器"""
        self.logger = logging.getLogger(__name__)
        self.profiles = {}
        
    def profile_function(self, func: Callable, *args, **kwargs):
        """
        分析函数性能
        
        参数:
            func: 要分析的函数
            *args, **kwargs: 函数参数
            
        返回:
            函数执行结果和性能信息
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            self.logger.error(f"函数执行失败: {e}")
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        profile_info = {
            'function_name': func.__name__,
            'execution_time_seconds': execution_time,
            'memory_delta_bytes': memory_delta,
            'memory_delta_mb': memory_delta / (1024**2),
            'success': success,
            'timestamp': datetime.now()
        }
        
        # 保存性能信息
        if func.__name__ not in self.profiles:
            self.profiles[func.__name__] = []
        self.profiles[func.__name__].append(profile_info)
        
        self.logger.info(f"函数 {func.__name__} 执行时间: {execution_time:.4f}秒, "
                        f"内存变化: {memory_delta / (1024**2):.2f}MB")
        
        return result, profile_info
        
    def get_function_stats(self, func_name: str) -> Dict:
        """
        获取函数性能统计
        
        参数:
            func_name: 函数名
            
        返回:
            性能统计信息
        """
        if func_name not in self.profiles:
            return {}
            
        profiles = self.profiles[func_name]
        if not profiles:
            return {}
            
        execution_times = [p['execution_time_seconds'] for p in profiles]
        memory_deltas = [p['memory_delta_mb'] for p in profiles]
        
        return {
            'function_name': func_name,
            'call_count': len(profiles),
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'std_execution_time': np.std(execution_times),
            'avg_memory_delta': np.mean(memory_deltas),
            'total_memory_delta': np.sum(memory_deltas),
            'success_rate': sum(1 for p in profiles if p['success']) / len(profiles)
        }
        
    def get_all_stats(self) -> pd.DataFrame:
        """
        获取所有函数的性能统计
        
        返回:
            性能统计DataFrame
        """
        stats = []
        for func_name in self.profiles:
            stat = self.get_function_stats(func_name)
            if stat:
                stats.append(stat)
                
        return pd.DataFrame(stats)

# 便捷函数
def get_system_info() -> Dict:
    """获取系统基本信息"""
    return {
        'platform': psutil.sys.platform,
        'python_version': psutil.sys.version,
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'disk_total_gb': psutil.disk_usage('/').total / (1024**3)
    }

def monitor_resource_usage(duration_seconds: int = 300, interval_seconds: int = 10) -> pd.DataFrame:
    """
    监控资源使用情况
    
    参数:
        duration_seconds: 监控持续时间
        interval_seconds: 监控间隔
        
    返回:
        资源使用数据DataFrame
    """
    monitor = SystemMonitor()
    metrics = []
    
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        metric = monitor.get_system_metrics()
        metrics.append(metric)
        time.sleep(interval_seconds)
        
    df = pd.DataFrame(metrics)
    df.set_index('timestamp', inplace=True)
    return df

# 示例使用
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建监控器
    monitor = SystemMonitor(log_interval=5)
    
    # 获取系统信息
    print("系统信息:", get_system_info())
    
    # 获取当前指标
    metrics = monitor.get_system_metrics()
    print("当前系统指标:", metrics)
    
    # 创建内存管理器
    memory_manager = MemoryManager()
    memory_usage = memory_manager.get_memory_usage()
    print("内存使用情况:", memory_usage)
    
    # 检查内存压力
    pressure = memory_manager.check_memory_pressure()
    print("内存压力评估:", pressure)
    
    # 创建性能分析器
    profiler = PerformanceProfiler()
    
    # 分析一个示例函数
    def sample_function():
        time.sleep(0.1)
        return [i for i in range(1000)]
    
    result, profile_info = profiler.profile_function(sample_function)
    print("函数性能分析:", profile_info) 