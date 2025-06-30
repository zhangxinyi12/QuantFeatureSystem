class FeatureValidator:
    def __init__(self, calc_timestamp):
        self.calc_timestamp = calc_timestamp
    
    def check_future_leakage(self, feature_df):
        """检查特征是否包含未来信息"""
        latest_date = feature_df.index.max()
        if latest_date >= self.calc_timestamp:
            raise FutureLeakageError(
                f"特征包含未来数据！最新日期: {latest_date}，计算时间: {self.calc_timestamp}"
            )
        
    def validate_feature_calc(self, feature_name, calculation_func):
        """装饰器：验证特征计算无未来信息"""
        def wrapper(data, *args, **kwargs):
            result = calculation_func(data, *args, **kwargs)
            self.check_future_leakage(result)
            return result
        return wrapper