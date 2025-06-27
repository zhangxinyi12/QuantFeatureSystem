#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日期处理工具模块
提供交易日历、日期转换、工作日计算等功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Union, Optional, Tuple
import warnings

class TradingCalendar:
    """交易日历类"""
    
    def __init__(self, market='CN'):
        """
        初始化交易日历
        
        参数:
            market: 市场代码 ('CN'=中国, 'US'=美国, 'HK'=香港)
        """
        self.market = market
        self._trading_days = None
        self._holidays = None
        
    def get_trading_days(self, start_date: Union[str, datetime], 
                        end_date: Union[str, datetime]) -> pd.DatetimeIndex:
        """
        获取指定时间范围内的交易日
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            交易日索引
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # 生成日期范围
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 过滤出工作日（周一到周五）
        trading_days = date_range[date_range.weekday < 5]
        
        # 移除节假日（这里简化处理，实际应用中需要完整的节假日数据）
        if self._holidays is not None:
            trading_days = trading_days[~trading_days.isin(self._holidays)]
            
        return trading_days
    
    def is_trading_day(self, date: Union[str, datetime]) -> bool:
        """
        判断是否为交易日
        
        参数:
            date: 日期
            
        返回:
            是否为交易日
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        # 检查是否为工作日
        if date.weekday() >= 5:  # 周六、周日
            return False
            
        # 检查是否为节假日
        if self._holidays is not None and date in self._holidays:
            return False
            
        return True
    
    def get_next_trading_day(self, date: Union[str, datetime]) -> datetime:
        """
        获取下一个交易日
        
        参数:
            date: 当前日期
            
        返回:
            下一个交易日
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        next_day = date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
            
        return next_day
    
    def get_previous_trading_day(self, date: Union[str, datetime]) -> datetime:
        """
        获取上一个交易日
        
        参数:
            date: 当前日期
            
        返回:
            上一个交易日
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        prev_day = date - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
            
        return prev_day
    
    def get_trading_days_between(self, start_date: Union[str, datetime], 
                                end_date: Union[str, datetime]) -> int:
        """
        计算两个日期之间的交易日数量
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            交易日数量
        """
        trading_days = self.get_trading_days(start_date, end_date)
        return len(trading_days)

def convert_to_trading_date(date: Union[str, datetime], 
                           calendar: TradingCalendar = None) -> datetime:
    """
    将任意日期转换为最近的交易日
    
    参数:
        date: 输入日期
        calendar: 交易日历对象
        
    返回:
        最近的交易日
    """
    if calendar is None:
        calendar = TradingCalendar()
        
    if isinstance(date, str):
        date = pd.to_datetime(date)
        
    if calendar.is_trading_day(date):
        return date
    else:
        return calendar.get_previous_trading_day(date)

def get_date_range_by_period(start_date: Union[str, datetime], 
                            period: str, 
                            calendar: TradingCalendar = None) -> Tuple[datetime, datetime]:
    """
    根据时间段获取日期范围
    
    参数:
        start_date: 开始日期
        period: 时间段 ('1d', '1w', '1m', '3m', '6m', '1y')
        calendar: 交易日历对象
        
    返回:
        (开始日期, 结束日期)的元组
    """
    if calendar is None:
        calendar = TradingCalendar()
        
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
        
    # 计算结束日期
    if period == '1d':
        end_date = start_date
    elif period == '1w':
        end_date = start_date + timedelta(days=7)
    elif period == '1m':
        end_date = start_date + pd.DateOffset(months=1)
    elif period == '3m':
        end_date = start_date + pd.DateOffset(months=3)
    elif period == '6m':
        end_date = start_date + pd.DateOffset(months=6)
    elif period == '1y':
        end_date = start_date + pd.DateOffset(years=1)
    else:
        raise ValueError(f"不支持的时间段: {period}")
    
    # 转换为交易日
    start_trading = convert_to_trading_date(start_date, calendar)
    end_trading = convert_to_trading_date(end_date, calendar)
    
    return start_trading, end_trading

def get_rolling_windows(start_date: Union[str, datetime], 
                       end_date: Union[str, datetime], 
                       window_size: int,
                       step_size: int = 1,
                       calendar: TradingCalendar = None) -> List[Tuple[datetime, datetime]]:
    """
    生成滚动窗口的日期范围
    
    参数:
        start_date: 开始日期
        end_date: 结束日期
        window_size: 窗口大小（交易日数）
        step_size: 步长（交易日数）
        calendar: 交易日历对象
        
    返回:
        日期范围列表
    """
    if calendar is None:
        calendar = TradingCalendar()
        
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # 获取交易日列表
    trading_days = calendar.get_trading_days(start_date, end_date)
    
    windows = []
    for i in range(0, len(trading_days) - window_size + 1, step_size):
        window_start = trading_days[i]
        window_end = trading_days[i + window_size - 1]
        windows.append((window_start, window_end))
    
    return windows

def format_date_for_query(date: Union[str, datetime]) -> str:
    """
    格式化日期为数据库查询格式
    
    参数:
        date: 日期
        
    返回:
        格式化后的日期字符串
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    return date.strftime('%Y-%m-%d')

def get_fiscal_period(date: Union[str, datetime]) -> Tuple[int, int]:
    """
    获取财年期间
    
    参数:
        date: 日期
        
    返回:
        (财年, 财季)的元组
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    year = date.year
    month = date.month
    
    # 假设财年从4月开始
    if month >= 4:
        fiscal_year = year
        fiscal_quarter = ((month - 4) // 3) + 1
    else:
        fiscal_year = year - 1
        fiscal_quarter = ((month + 8) // 3) + 1
    
    return fiscal_year, fiscal_quarter

def is_month_end(date: Union[str, datetime]) -> bool:
    """
    判断是否为月末
    
    参数:
        date: 日期
        
    返回:
        是否为月末
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    next_day = date + timedelta(days=1)
    return date.month != next_day.month

def is_quarter_end(date: Union[str, datetime]) -> bool:
    """
    判断是否为季末
    
    参数:
        date: 日期
        
    返回:
        是否为季末
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    return date.month in [3, 6, 9, 12] and is_month_end(date)

def is_year_end(date: Union[str, datetime]) -> bool:
    """
    判断是否为年末
    
    参数:
        date: 日期
        
    返回:
        是否为年末
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    return date.month == 12 and is_month_end(date)

# 便捷函数
def get_current_trading_date(calendar: TradingCalendar = None) -> datetime:
    """获取当前交易日"""
    if calendar is None:
        calendar = TradingCalendar()
    
    current_date = datetime.now()
    return convert_to_trading_date(current_date, calendar)

def get_trading_days_count(start_date: Union[str, datetime], 
                          end_date: Union[str, datetime],
                          calendar: TradingCalendar = None) -> int:
    """获取两个日期之间的交易日数量"""
    if calendar is None:
        calendar = TradingCalendar()
    
    return calendar.get_trading_days_between(start_date, end_date)

# 示例使用
if __name__ == "__main__":
    # 创建交易日历
    calendar = TradingCalendar()
    
    # 测试基本功能
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    print(f"交易日数量: {get_trading_days_count(start_date, end_date, calendar)}")
    print(f"当前交易日: {get_current_trading_date(calendar)}")
    
    # 测试滚动窗口
    windows = get_rolling_windows(start_date, end_date, window_size=20, step_size=5)
    print(f"生成了 {len(windows)} 个滚动窗口")
    
    # 测试时间段
    start, end = get_date_range_by_period("2023-01-01", "3m", calendar)
    print(f"3个月时间段: {start} 到 {end}") 