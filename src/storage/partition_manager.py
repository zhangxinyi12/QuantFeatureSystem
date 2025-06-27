class TimePartitioner:
    def __init__(self, partition_interval='monthly'):
        self.interval = partition_interval
    
    def create_partition_ddl(self, table_name, date_column):
        """生成分区DDL语句"""
        if self.interval == 'daily':
            return f"PARTITION BY RANGE ({date_column})"
        elif self.interval == 'monthly':
            return f"PARTITION BY RANGE (EXTRACT(YEAR FROM {date_column}), EXTRACT(MONTH FROM {date_column}))"
        else:
            return ""

    def get_partition_filter(self, start_date, end_date):
        """获取分区查询条件"""
        partitions = []
        current = start_date
        while current <= end_date:
            if self.interval == 'monthly':
                partition = f"features_{current.year}_{current.month:02d}"
                partitions.append(partition)
                current += relativedelta(months=1)
        return " OR ".join([f"partition = '{p}'" for p in partitions])
