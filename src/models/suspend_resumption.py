#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
停牌复牌表数据模型
LC_SuspendResumption - 上市公司/基金/债券停牌复牌信息
"""

from datetime import datetime, date
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd


class InfoSource(Enum):
    """信息来源枚举 - 对应CT_SystemConst表中LB=201的DM值"""
    BEIJING_EXCHANGE = 18   # 北京证券交易所
    SHANGHAI_EXCHANGE = 83  # 上海证券交易所
    SHENZHEN_EXCHANGE = 90  # 深圳证券交易所


class SuspendType(Enum):
    """停牌期限类型枚举 - 对应CT_SystemConst表中的DM值"""
    MORNING_SUSPEND = 10      # 上午停牌
    AFTERNOON_SUSPEND = 20    # 下午停牌
    CONTINUOUS_SUSPEND = 30   # 今起连续停牌
    INTRA_DAY_SUSPEND = 40    # 盘中停牌
    ONE_DAY_SUSPEND = 50      # 停牌1天
    ONE_HOUR_SUSPEND = 60     # 停牌1小时


class SuspendStatement(Enum):
    """停牌事项说明枚举 - 对应CT_SystemConst表中LB=1654的DM值"""
    TEMPORARY_SUSPEND = 101           # 临时停牌
    SHAREHOLDERS_MEETING = 102        # 召开股东大会
    MAJOR_EVENT = 103                 # 重大事项
    OTHER_ANNOUNCEMENT = 104          # 其它公告（停牌）
    ABNORMAL_FLUCTUATION = 105        # 交易异常波动
    CLARIFICATION = 106               # 澄清公告
    CANCEL_SPECIAL_TREATMENT = 107    # 撤销其他特别处理公告
    INTRA_DAY_TEMPORARY = 108         # 盘中临时停牌
    CANCEL_DELISTING_RISK = 109       # 撤销退市风险警示公告
    FAILED_RESOLUTION = 110           # 未能如期刊登股东大会决议
    ADDITIONAL_ISSUE_NOTICE = 111     # 增发提示性公告
    CONTINUOUS_ISSUE_BID = 112        # 续发行招投标
    PRICE_ABNORMAL = 113              # 股价异动停牌公告
    SHARE_SUSPEND = 114               # 份额暂停交易公告
    TRADING_RISK = 115                # 交易风险提示
    INCOME_DISTRIBUTION = 116         # 收益分配
    DELISTING_RISK_WARNING = 117      # 实行退市风险警示公告
    SPECIAL_TREATMENT = 118           # 实行其他特别处理公告
    FAILED_PERIODIC_REPORT = 119      # 未按期披露定期报告
    BANKRUPTCY = 120                  # 破产
    TERMINATE_LISTING = 121           # 拟终止挂牌
    INSUFFICIENT_MAKERS = 122         # 做市商不足2家
    TRANSFER_LISTING = 123            # 转板上市
    IMPORTANT_ANNOUNCEMENT = 603      # 刊登重要公告
    MAJOR_ASSET_RESTRUCTURE = 604     # 拟筹划重大资产重组
    IMPORTANT_EVENT_UNANNOUNCED = 605 # 重要事项未公告
    FAILED_RESOLUTION_ANNOUNCEMENT = 606 # 未刊登股东大会决议公告
    ABNORMAL_FLUCTUATION_ANNOUNCEMENT = 607 # 刊登股票交易异常波动公告
    MEDIA_CLARIFICATION = 608         # 媒体报道需澄清
    FUND_COMPANY_APPLICATION = 610    # 基金公司申请
    PRICING_ADDITIONAL_ISSUE = 611    # 定价增发
    UNDERLYING_STOCK_SUSPEND = 612    # 正股停牌
    OTHER_SPECIAL_REASON = 999        # 其他特别原因


@dataclass
class SuspendResumption:
    """
    停牌复牌信息数据模型
    
    业务唯一性：InnerCode,SuspendDate,SuspendTime
    数据范围：2008.04-至今
    更新频率：日更新
    """
    
    # 基础字段
    id: int
    inner_code: int
    info_publ_date: date
    info_source: int
    suspend_date: date
    suspend_time: str
    suspend_reason: str
    suspend_statement: int
    suspend_term: Optional[str] = None
    suspend_type: int
    resumption_date: Optional[date] = None
    resumption_time: Optional[str] = None
    resumption_statement: Optional[str] = None
    insert_time: date
    update_time: date
    jsid: int
    
    def __post_init__(self):
        """数据验证和转换"""
        # 确保日期字段为date类型
        if isinstance(self.info_publ_date, str):
            self.info_publ_date = datetime.strptime(self.info_publ_date, '%Y-%m-%d').date()
        if isinstance(self.suspend_date, str):
            self.suspend_date = datetime.strptime(self.suspend_date, '%Y-%m-%d').date()
        if isinstance(self.resumption_date, str) and self.resumption_date:
            self.resumption_date = datetime.strptime(self.resumption_date, '%Y-%m-%d').date()
        if isinstance(self.insert_time, str):
            self.insert_time = datetime.strptime(self.insert_time, '%Y-%m-%d').date()
        if isinstance(self.update_time, str):
            self.update_time = datetime.strptime(self.update_time, '%Y-%m-%d').date()
    
    @property
    def info_source_name(self) -> str:
        """获取信息来源名称"""
        # 使用InfoSourceManager的默认映射
        source_manager = InfoSourceManager()
        return source_manager.get_description(self.info_source)
    
    @property
    def suspend_statement_name(self) -> str:
        """获取停牌事项说明名称"""
        # 使用SuspendStatementManager的默认映射
        statement_manager = SuspendStatementManager()
        return statement_manager.get_description(self.suspend_statement)
    
    @property
    def suspend_type_name(self) -> str:
        """获取停牌期限类型名称"""
        # 使用SuspendTypeManager的默认映射
        type_manager = SuspendTypeManager()
        return type_manager.get_description(self.suspend_type)
    
    @property
    def is_suspended(self) -> bool:
        """是否正在停牌"""
        if not self.resumption_date:
            return True
        return self.resumption_date > date.today()
    
    @property
    def suspend_duration_days(self) -> Optional[int]:
        """停牌持续天数"""
        if self.resumption_date and self.suspend_date:
            return (self.resumption_date - self.suspend_date).days
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'inner_code': self.inner_code,
            'info_publ_date': self.info_publ_date,
            'info_source': self.info_source,
            'info_source_name': self.info_source_name,
            'suspend_date': self.suspend_date,
            'suspend_time': self.suspend_time,
            'suspend_reason': self.suspend_reason,
            'suspend_statement': self.suspend_statement,
            'suspend_statement_name': self.suspend_statement_name,
            'suspend_term': self.suspend_term,
            'suspend_type': self.suspend_type,
            'suspend_type_name': self.suspend_type_name,
            'resumption_date': self.resumption_date,
            'resumption_time': self.resumption_time,
            'resumption_statement': self.resumption_statement,
            'insert_time': self.insert_time,
            'update_time': self.update_time,
            'jsid': self.jsid,
            'is_suspended': self.is_suspended,
            'suspend_duration_days': self.suspend_duration_days
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuspendResumption':
        """从字典创建实例"""
        return cls(**data)
    
    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> 'SuspendResumption':
        """从DataFrame行创建实例"""
        return cls(
            id=row['ID'],
            inner_code=row['InnerCode'],
            info_publ_date=row['InfoPublDate'],
            info_source=row['InfoSource'],
            suspend_date=row['SuspendDate'],
            suspend_time=row['SuspendTime'],
            suspend_reason=row['SuspendReason'],
            suspend_statement=row['SuspendStatement'],
            suspend_term=row.get('SuspendTerm'),
            suspend_type=row['SuspendType'],
            resumption_date=row.get('ResumptionDate'),
            resumption_time=row.get('ResumptionTime'),
            resumption_statement=row.get('ResumptionStatement'),
            insert_time=row['InsertTime'],
            update_time=row['UpdateTime'],
            jsid=row['JSID']
        )


class SuspendResumptionManager:
    """停牌复牌信息管理器"""
    
    def __init__(self):
        self.records: Dict[str, SuspendResumption] = {}
    
    def add_record(self, record: SuspendResumption):
        """添加记录"""
        key = f"{record.inner_code}_{record.suspend_date}_{record.suspend_time}"
        self.records[key] = record
    
    def get_by_inner_code(self, inner_code: int) -> list[SuspendResumption]:
        """根据内部编码获取记录"""
        return [r for r in self.records.values() if r.inner_code == inner_code]
    
    def get_current_suspensions(self) -> list[SuspendResumption]:
        """获取当前正在停牌的记录"""
        return [r for r in self.records.values() if r.is_suspended]
    
    def get_suspensions_by_date_range(self, start_date: date, end_date: date) -> list[SuspendResumption]:
        """根据日期范围获取停牌记录"""
        return [r for r in self.records.values() 
                if start_date <= r.suspend_date <= end_date]
    
    def get_suspensions_by_reason(self, reason_keyword: str) -> list[SuspendResumption]:
        """根据停牌原因关键词获取记录"""
        return [r for r in self.records.values() 
                if reason_keyword.lower() in r.suspend_reason.lower()]
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        if not self.records:
            return pd.DataFrame()
        
        data = [record.to_dict() for record in self.records.values()]
        return pd.DataFrame(data)
    
    def load_from_dataframe(self, df: pd.DataFrame):
        """从DataFrame加载数据"""
        for _, row in df.iterrows():
            record = SuspendResumption.from_dataframe_row(row)
            self.add_record(record)


class SuspendTypeManager:
    """停牌期限类型管理器 - 管理与CT_SystemConst表的映射"""
    
    def __init__(self):
        self.type_mappings = {
            10: "上午停牌",
            20: "下午停牌", 
            30: "今起连续停牌",
            40: "盘中停牌",
            50: "停牌1天",
            60: "停牌1小时"
        }
    
    def get_description(self, dm: int) -> str:
        """根据DM值获取描述"""
        return self.type_mappings.get(dm, f"未知类型({dm})")
    
    def get_dm_by_description(self, description: str) -> Optional[int]:
        """根据描述获取DM值"""
        for dm, desc in self.type_mappings.items():
            if desc == description:
                return dm
        return None
    
    def load_from_database(self, db_connection):
        """从数据库加载停牌期限类型映射"""
        try:
            query = SuspendResumptionQueries.get_suspend_type_descriptions()
            result = db_connection.query(query)
            
            self.type_mappings = {}
            for row in result:
                self.type_mappings[row['DM']] = row['MC']
                
        except Exception as e:
            print(f"加载停牌期限类型映射失败: {e}")
            # 使用默认映射
    
    def get_all_types(self) -> Dict[int, str]:
        """获取所有停牌期限类型"""
        return self.type_mappings.copy()


class SuspendStatementManager:
    """停牌事项说明管理器 - 管理与CT_SystemConst表的映射"""
    
    def __init__(self):
        self.statement_mappings = {
            101: "临时停牌",
            102: "召开股东大会",
            103: "重大事项",
            104: "其它公告（停牌）",
            105: "交易异常波动",
            106: "澄清公告",
            107: "撤销其他特别处理公告",
            108: "盘中临时停牌",
            109: "撤销退市风险警示公告",
            110: "未能如期刊登股东大会决议",
            111: "增发提示性公告",
            112: "续发行招投标",
            113: "股价异动停牌公告",
            114: "份额暂停交易公告",
            115: "交易风险提示",
            116: "收益分配",
            117: "实行退市风险警示公告",
            118: "实行其他特别处理公告",
            119: "未按期披露定期报告",
            120: "破产",
            121: "拟终止挂牌",
            122: "做市商不足2家",
            123: "转板上市",
            603: "刊登重要公告",
            604: "拟筹划重大资产重组",
            605: "重要事项未公告",
            606: "未刊登股东大会决议公告",
            607: "刊登股票交易异常波动公告",
            608: "媒体报道需澄清",
            610: "基金公司申请",
            611: "定价增发",
            612: "正股停牌",
            999: "其他特别原因"
        }
    
    def get_description(self, dm: int) -> str:
        """根据DM值获取描述"""
        return self.statement_mappings.get(dm, f"未知事项({dm})")
    
    def get_dm_by_description(self, description: str) -> Optional[int]:
        """根据描述获取DM值"""
        for dm, desc in self.statement_mappings.items():
            if desc == description:
                return dm
        return None
    
    def load_from_database(self, db_connection):
        """从数据库加载停牌事项说明映射"""
        try:
            query = SuspendResumptionQueries.get_suspend_statement_descriptions()
            result = db_connection.query(query)
            
            self.statement_mappings = {}
            for row in result:
                self.statement_mappings[row['DM']] = row['MC']
                
        except Exception as e:
            print(f"加载停牌事项说明映射失败: {e}")
            # 使用默认映射
    
    def get_all_statements(self) -> Dict[int, str]:
        """获取所有停牌事项说明"""
        return self.statement_mappings.copy()


class InfoSourceManager:
    """信息来源管理器 - 管理与CT_SystemConst表的映射"""
    
    def __init__(self):
        self.source_mappings = {
            18: "北京证券交易所",
            83: "上海证券交易所", 
            90: "深圳证券交易所"
        }
    
    def get_description(self, dm: int) -> str:
        """根据DM值获取描述"""
        return self.source_mappings.get(dm, f"未知来源({dm})")
    
    def get_dm_by_description(self, description: str) -> Optional[int]:
        """根据描述获取DM值"""
        for dm, desc in self.source_mappings.items():
            if desc == description:
                return dm
        return None
    
    def load_from_database(self, db_connection):
        """从数据库加载信息来源映射"""
        try:
            query = SuspendResumptionQueries.get_info_source_descriptions()
            result = db_connection.query(query)
            
            self.source_mappings = {}
            for row in result:
                self.source_mappings[row['DM']] = row['MC']
                
        except Exception as e:
            print(f"加载信息来源映射失败: {e}")
            # 使用默认映射
            self.source_mappings = {
                18: "北京证券交易所",
                83: "上海证券交易所", 
                90: "深圳证券交易所"
            }
    
    def get_all_sources(self) -> Dict[int, str]:
        """获取所有信息来源"""
        return self.source_mappings.copy()


# 数据库查询相关
class SuspendResumptionQueries:
    """停牌复牌表查询语句"""
    
    @staticmethod
    def get_info_source_descriptions() -> str:
        """获取信息来源描述 - 从CT_SystemConst表"""
        return """
        SELECT DM, MC 
        FROM CT_SystemConst 
        WHERE LB = 201 AND DM IN (18, 83, 90)
        ORDER BY DM
        """
    
    @staticmethod
    def get_suspend_statement_descriptions() -> str:
        """获取停牌事项说明描述 - 从CT_SystemConst表"""
        return """
        SELECT DM, MC 
        FROM CT_SystemConst 
        WHERE LB = 1654
        ORDER BY DM
        """
    
    @staticmethod
    def get_suspend_type_descriptions() -> str:
        """获取停牌期限类型描述 - 从CT_SystemConst表"""
        return """
        SELECT DM, MC 
        FROM CT_SystemConst 
        WHERE LB = [需要确认LB值] AND DM IN (10, 20, 30, 40, 50, 60)
        ORDER BY DM
        """
    
    @staticmethod
    def get_table_info() -> str:
        """获取表信息"""
        return """
        SELECT 
            COUNT(*) as total_records,
            MIN(SuspendDate) as earliest_date,
            MAX(SuspendDate) as latest_date,
            COUNT(DISTINCT InnerCode) as unique_securities
        FROM LC_SuspendResumption
        """
    
    @staticmethod
    def get_by_inner_code(inner_code: int) -> str:
        """根据内部编码查询"""
        return f"""
        SELECT * FROM LC_SuspendResumption 
        WHERE InnerCode = {inner_code}
        ORDER BY SuspendDate DESC, SuspendTime DESC
        """
    
    @staticmethod
    def get_current_suspensions() -> str:
        """获取当前正在停牌的记录"""
        return """
        SELECT * FROM LC_SuspendResumption 
        WHERE ResumptionDate IS NULL OR ResumptionDate > CURDATE()
        ORDER BY SuspendDate DESC
        """
    
    @staticmethod
    def get_by_date_range(start_date: str, end_date: str) -> str:
        """根据日期范围查询"""
        return f"""
        SELECT * FROM LC_SuspendResumption 
        WHERE SuspendDate BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY SuspendDate DESC, SuspendTime DESC
        """
    
    @staticmethod
    def get_by_reason_keyword(keyword: str) -> str:
        """根据停牌原因关键词查询"""
        return f"""
        SELECT * FROM LC_SuspendResumption 
        WHERE SuspendReason LIKE '%{keyword}%'
        ORDER BY SuspendDate DESC
        """
    
    @staticmethod
    def get_statistics_by_source() -> str:
        """按信息来源统计"""
        return """
        SELECT 
            InfoSource,
            COUNT(*) as count,
            COUNT(DISTINCT InnerCode) as unique_securities
        FROM LC_SuspendResumption
        GROUP BY InfoSource
        ORDER BY count DESC
        """
    
    @staticmethod
    def get_statistics_by_type() -> str:
        """按停牌类型统计"""
        return """
        SELECT 
            SuspendType,
            COUNT(*) as count,
            AVG(CASE WHEN ResumptionDate IS NOT NULL 
                THEN DATEDIFF(ResumptionDate, SuspendDate) 
                ELSE NULL END) as avg_duration_days
        FROM LC_SuspendResumption
        GROUP BY SuspendType
        ORDER BY count DESC
        """


if __name__ == "__main__":
    # 测试代码
    print("=== 停牌复牌表数据模型测试 ===")
    
    # 创建测试实例
    test_record = SuspendResumption(
        id=1,
        inner_code=100001,
        info_publ_date=date(2024, 1, 15),
        info_source=1,
        suspend_date=date(2024, 1, 16),
        suspend_time="09:30:00",
        suspend_reason="重大事项停牌",
        suspend_statement=1,
        suspend_term="不超过5个交易日",
        suspend_type=1,
        resumption_date=date(2024, 1, 20),
        resumption_time="09:30:00",
        resumption_statement="复牌公告",
        insert_time=date(2024, 1, 15),
        update_time=date(2024, 1, 15),
        jsid=1001
    )
    
    print(f"测试记录: {test_record.to_dict()}")
    print(f"信息来源: {test_record.info_source_name}")
    print(f"停牌类型: {test_record.suspend_type_name}")
    print(f"是否停牌: {test_record.is_suspended}")
    print(f"停牌天数: {test_record.suspend_duration_days}") 