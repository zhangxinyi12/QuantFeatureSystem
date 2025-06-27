class MomentumFeatures:
    def __init__(self, lookback_period=21):
        self.lookback = lookback_period
    
    @FeatureValidator(Config.CALC_TIMESTAMP).validate_feature_calc
    def calculate(self, data):
        """计算动量特征，严格使用历史数据"""
        return data['adj_close'].pct_change(self.lookback)

    @property
    def description(self):
        return f"{self.lookback}日价格动量，反映中期趋势强度"
