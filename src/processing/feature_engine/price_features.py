"""
量化特征工程模块
包含价格衍生特征、成交量衍生特征、技术指标等
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger('feature_engine')

class QuantFeatureEngine:
    """量化特征工程引擎"""
    
    def __init__(self):
        """初始化特征工程引擎"""
        self.feature_groups = {
            'price': ['returns', 'momentum', 'volatility', 'trend', 'support_resistance'],
            'volume': ['volume_indicators', 'volume_price', 'volume_patterns'],
            'technical': ['oscillators', 'patterns', 'seasonality'],
            'cross_asset': ['spreads', 'ratios', 'correlations']
        }
    
    def calculate_all_features(self, df: pd.DataFrame, feature_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算所有特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            feature_types: 要计算的特征类型列表
            
        Returns:
            pd.DataFrame: 包含所有特征的DataFrame
        """
        if feature_types is None:
            feature_types = ['price', 'volume', 'technical']
        
        result_df = df.copy()
        
        for feature_type in feature_types:
            if feature_type == 'price':
                result_df = self.calculate_price_features(result_df)
            elif feature_type == 'volume':
                result_df = self.calculate_volume_features(result_df)
            elif feature_type == 'technical':
                result_df = self.calculate_technical_features(result_df)
            elif feature_type == 'cross_asset':
                result_df = self.calculate_cross_asset_features(result_df)
        
        logger.info(f"特征计算完成，共生成 {len(result_df.columns)} 个特征")
        return result_df

    # ==================== 价格衍生特征 ====================
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格衍生特征"""
        df = self.calculate_returns(df)
        df = self.calculate_momentum(df)
        df = self.calculate_volatility(df)
        df = self.calculate_trend_indicators(df)
        df = self.calculate_support_resistance(df)
        return df
    
    def calculate_returns(self, df: pd.DataFrame, periods: List[int] = [1, 5, 20, 60, 252]) -> pd.DataFrame:
        """
        计算收益率特征
        
        Args:
            df: 包含Close价格的DataFrame
            periods: 收益率计算周期列表
            
        Returns:
            pd.DataFrame: 包含收益率特征的DataFrame
        """
        df = df.copy()
        close = df['ClosePrice'] if 'ClosePrice' in df.columns else df['Close']
        
        for period in periods:
            # 绝对收益率
            df[f'Return_{period}d'] = close.pct_change(period)
            
            # 对数收益率
            df[f'LogReturn_{period}d'] = np.log(close / close.shift(period))
            
            # 累积收益率
            df[f'CumReturn_{period}d'] = (close / close.shift(period)) - 1
        
        return df
    
    def calculate_momentum(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        计算动量特征
        
        Args:
            df: 包含价格数据的DataFrame
            periods: 动量计算周期列表
            
        Returns:
            pd.DataFrame: 包含动量特征的DataFrame
        """
        df = df.copy()
        close = df['ClosePrice'] if 'ClosePrice' in df.columns else df['Close']
        
        for period in periods:
            # 简单动量
            df[f'Momentum_{period}d'] = close - close.shift(period)
            
            # 价格变化率
            df[f'ROC_{period}d'] = (close / close.shift(period) - 1) * 100
            
            # 相对动量（相对于移动平均）
            sma = close.rolling(window=period).mean()
            df[f'RelMomentum_{period}d'] = (close - sma) / sma
        
        return df
    
    def calculate_volatility(self, df: pd.DataFrame, windows: List[int] = [20, 60, 252]) -> pd.DataFrame:
        """
        计算波动率特征
        
        Args:
            df: 包含OHLC数据的DataFrame
            windows: 波动率计算窗口列表
            
        Returns:
            pd.DataFrame: 包含波动率特征的DataFrame
        """
        df = df.copy()
        close = df['ClosePrice'] if 'ClosePrice' in df.columns else df['Close']
        high = df['HighPrice'] if 'HighPrice' in df.columns else df['High']
        low = df['LowPrice'] if 'LowPrice' in df.columns else df['Low']
        open_price = df['OpenPrice'] if 'OpenPrice' in df.columns else df['Open']
        
        # 计算对数收益率
        log_returns = np.log(close / close.shift(1))
        
        for window in windows:
            # 历史波动率（标准差）
            df[f'HistVol_{window}d'] = log_returns.rolling(window=window).std() * np.sqrt(252)
            
            # 已实现波动率
            df[f'RealizedVol_{window}d'] = np.sqrt(
                (log_returns ** 2).rolling(window=window).sum() * 252 / window
            )
            
            # Parkinson波动率（利用高低价信息）
            hl_ratio = np.log(high / low)
            df[f'ParkinsonVol_{window}d'] = np.sqrt(
                (1 / (4 * window * np.log(2))) * 
                (hl_ratio ** 2).rolling(window=window).sum() * 252
            )
            
            # Garman-Klass波动率
            log_hl = np.log(high / low)
            log_co = np.log(close / open_price)
            gk_vol = np.sqrt(
                (0.5 * log_hl ** 2) - ((2 * np.log(2) - 1) * log_co ** 2)
            )
            df[f'GKVol_{window}d'] = gk_vol.rolling(window=window).mean() * np.sqrt(252)
            
            # 平均真实波幅（ATR）
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'ATR_{window}d'] = tr.rolling(window=window).mean()
        
        return df
    
    def calculate_trend_indicators(self, df: pd.DataFrame, windows: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """
        计算趋势指标
        
        Args:
            df: 包含价格数据的DataFrame
            windows: 移动平均窗口列表
            
        Returns:
            pd.DataFrame: 包含趋势指标的DataFrame
        """
        df = df.copy()
        close = df['ClosePrice'] if 'ClosePrice' in df.columns else df['Close']
        
        for window in windows:
            # 简单移动平均
            df[f'SMA_{window}'] = close.rolling(window=window).mean()
            
            # 指数移动平均
            df[f'EMA_{window}'] = close.ewm(span=window, adjust=False).mean()
            
            # 价格相对位置
            df[f'PricePos_{window}'] = (close - df[f'SMA_{window}']) / df[f'SMA_{window}']
        
        # 布林带
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        df['BB_Upper'] = sma_20 + (2 * std_20)
        df['BB_Lower'] = sma_20 - (2 * std_20)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma_20
        df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    def calculate_support_resistance(self, df: pd.DataFrame, windows: List[int] = [20, 60, 252]) -> pd.DataFrame:
        """
        计算支撑阻力位
        
        Args:
            df: 包含高低价数据的DataFrame
            windows: 支撑阻力计算窗口列表
            
        Returns:
            pd.DataFrame: 包含支撑阻力特征的DataFrame
        """
        df = df.copy()
        high = df['HighPrice'] if 'HighPrice' in df.columns else df['High']
        low = df['LowPrice'] if 'LowPrice' in df.columns else df['Low']
        close = df['ClosePrice'] if 'ClosePrice' in df.columns else df['Close']
        
        for window in windows:
            # 支撑位（近期最低价）
            df[f'Support_{window}d'] = low.rolling(window=window).min()
            
            # 阻力位（近期最高价）
            df[f'Resistance_{window}d'] = high.rolling(window=window).max()
            
            # 价格位置
            df[f'PricePosition_{window}d'] = (
                (close - df[f'Support_{window}d']) / 
                (df[f'Resistance_{window}d'] - df[f'Support_{window}d'])
            )
        
        # 斐波那契回撤位
        swing_high = high.rolling(window=20).max()
        swing_low = low.rolling(window=20).min()
        delta = swing_high - swing_low
        
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in fib_levels:
            df[f'Fib_{level}'] = swing_high - level * delta
        
        return df

    # ==================== 成交量衍生特征 ====================
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量衍生特征"""
        df = self.calculate_volume_indicators(df)
        df = self.calculate_volume_price_features(df)
        df = self.calculate_volume_patterns(df)
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame, windows: List[int] = [5, 20, 60]) -> pd.DataFrame:
        """
        计算成交量指标
        
        Args:
            df: 包含成交量数据的DataFrame
            windows: 成交量计算窗口列表
            
        Returns:
            pd.DataFrame: 包含成交量指标的DataFrame
        """
        df = df.copy()
        volume = df['TurnoverVolume'] if 'TurnoverVolume' in df.columns else df['Volume']
        
        for window in windows:
            # 成交量移动平均
            df[f'Volume_MA_{window}'] = volume.rolling(window=window).mean()
            
            # 成交量比率
            df[f'Volume_Ratio_{window}'] = volume / df[f'Volume_MA_{window}']
            
            # 成交量标准差
            df[f'Volume_Std_{window}'] = volume.rolling(window=window).std()
            
            # 成交量变化率
            df[f'Volume_Change_{window}'] = volume.pct_change(window)
        
        # 成交量加权平均价格（VWAP）
        typical_price = (df['HighPrice'] + df['LowPrice'] + df['ClosePrice']) / 3
        df['VWAP'] = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        
        return df
    
    def calculate_volume_price_features(self, df: pd.DataFrame, windows: List[int] = [5, 20]) -> pd.DataFrame:
        """
        计算价量关系特征
        
        Args:
            df: 包含价格和成交量数据的DataFrame
            windows: 计算窗口列表
            
        Returns:
            pd.DataFrame: 包含价量关系特征的DataFrame
        """
        df = df.copy()
        close = df['ClosePrice'] if 'ClosePrice' in df.columns else df['Close']
        volume = df['TurnoverVolume'] if 'TurnoverVolume' in df.columns else df['Volume']
        
        # 价格变化
        price_change = close.pct_change()
        
        for window in windows:
            # 价量相关性
            df[f'PriceVolume_Corr_{window}'] = (
                price_change.rolling(window=window)
                .corr(volume.rolling(window=window))
            )
            
            # 价量背离指标
            price_ma = close.rolling(window=window).mean()
            volume_ma = volume.rolling(window=window).mean()
            
            df[f'PriceVolume_Divergence_{window}'] = (
                (close - price_ma) / price_ma - (volume - volume_ma) / volume_ma
            )
            
            # 成交量加权价格动量
            vwap = (close * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
            df[f'VWAP_Momentum_{window}'] = (close - vwap) / vwap
        
        # 资金流量指标（MFI）
        typical_price = (df['HighPrice'] + df['LowPrice'] + df['ClosePrice']) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        
        df['MFI'] = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return df
    
    def calculate_volume_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量模式特征
        
        Args:
            df: 包含成交量数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含成交量模式特征的DataFrame
        """
        df = df.copy()
        volume = df['TurnoverVolume'] if 'TurnoverVolume' in df.columns else df['Volume']
        
        # 成交量异常检测
        volume_ma = volume.rolling(window=20).mean()
        volume_std = volume.rolling(window=20).std()
        
        df['Volume_ZScore'] = (volume - volume_ma) / volume_std
        df['Volume_Spike'] = (df['Volume_ZScore'] > 2).astype(int)
        
        # 成交量趋势
        df['Volume_Trend'] = volume.rolling(window=5).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
        )
        
        # 成交量分布特征
        df['Volume_Percentile'] = volume.rolling(window=252).rank(pct=True)
        
        return df

    # ==================== 技术指标特征 ====================
    
    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        df = self.calculate_oscillators(df)
        df = self.calculate_patterns(df)
        df = self.calculate_seasonality(df)
        return df
    
    def calculate_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算震荡指标
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含震荡指标的DataFrame
        """
        df = df.copy()
        close = df['ClosePrice'] if 'ClosePrice' in df.columns else df['Close']
        high = df['HighPrice'] if 'HighPrice' in df.columns else df['High']
        low = df['LowPrice'] if 'LowPrice' in df.columns else df['Low']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 随机指标（Stochastic）
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        df['Stoch_K'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # 威廉指标（Williams %R）
        df['Williams_R'] = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return df
    
    def calculate_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术形态特征
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含技术形态特征的DataFrame
        """
        df = df.copy()
        close = df['ClosePrice'] if 'ClosePrice' in df.columns else df['Close']
        high = df['HighPrice'] if 'HighPrice' in df.columns else df['High']
        low = df['LowPrice'] if 'LowPrice' in df.columns else df['Low']
        open_price = df['OpenPrice'] if 'OpenPrice' in df.columns else df['Open']
        
        # 蜡烛图形态
        body_size = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        
        # 锤子线
        df['Hammer'] = (
            (lower_shadow > 2 * body_size) & 
            (upper_shadow < body_size) & 
            (close > open_price)
        ).astype(int)
        
        # 上吊线
        df['Hanging_Man'] = (
            (lower_shadow > 2 * body_size) & 
            (upper_shadow < body_size) & 
            (close < open_price)
        ).astype(int)
        
        # 十字星
        df['Doji'] = (body_size < 0.1 * (high - low)).astype(int)
        
        # 吞没形态
        df['Bullish_Engulfing'] = (
            (close.shift(1) < open_price.shift(1)) &  # 前一根为阴线
            (close > open_price) &  # 当前为阳线
            (open_price < close.shift(1)) &  # 开盘价低于前一根收盘价
            (close > open_price.shift(1))  # 收盘价高于前一根开盘价
        ).astype(int)
        
        df['Bearish_Engulfing'] = (
            (close.shift(1) > open_price.shift(1)) &  # 前一根为阳线
            (close < open_price) &  # 当前为阴线
            (open_price > close.shift(1)) &  # 开盘价高于前一根收盘价
            (close < open_price.shift(1))  # 收盘价低于前一根开盘价
        ).astype(int)
        
        return df
    
    def calculate_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算季节性特征
        
        Args:
            df: 包含时间索引的DataFrame
            
        Returns:
            pd.DataFrame: 包含季节性特征的DataFrame
        """
        df = df.copy()
        
        # 确保索引是时间类型
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'TradingDay' in df.columns:
                df['TradingDay'] = pd.to_datetime(df['TradingDay'])
                df.set_index('TradingDay', inplace=True)
        
        # 时间特征
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfYear'] = df.index.dayofyear
        
        # 季节性指标
        close = df['ClosePrice'] if 'ClosePrice' in df.columns else df['Close']
        
        # 月度季节性
        monthly_returns = close.pct_change().groupby(df.index.month).mean()
        df['Monthly_Seasonality'] = df.index.month.map(monthly_returns)
        
        # 周内季节性
        weekly_returns = close.pct_change().groupby(df.index.dayofweek).mean()
        df['Weekly_Seasonality'] = df.index.dayofweek.map(weekly_returns)
        
        return df

    # ==================== 跨资产特征 ====================
    
    def calculate_cross_asset_features(self, df: pd.DataFrame, other_assets: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算跨资产特征
        
        Args:
            df: 主资产数据
            other_assets: 其他资产数据字典
            
        Returns:
            pd.DataFrame: 包含跨资产特征的DataFrame
        """
        df = df.copy()
        
        if other_assets is not None:
            for asset_name, asset_data in other_assets.items():
                # 价差
                df[f'Spread_vs_{asset_name}'] = df['ClosePrice'] - asset_data['ClosePrice']
                
                # 比率
                df[f'Ratio_vs_{asset_name}'] = df['ClosePrice'] / asset_data['ClosePrice']
                
                # 相关性
                df[f'Corr_vs_{asset_name}'] = (
                    df['ClosePrice'].rolling(window=20)
                    .corr(asset_data['ClosePrice'].rolling(window=20))
                )
        
        return df

# 便捷函数
def calculate_all_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """计算所有特征的便捷函数"""
    engine = QuantFeatureEngine()
    return engine.calculate_all_features(df, **kwargs)

def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算价格特征的便捷函数"""
    engine = QuantFeatureEngine()
    return engine.calculate_price_features(df)

def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算成交量特征的便捷函数"""
    engine = QuantFeatureEngine()
    return engine.calculate_volume_features(df)

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标的便捷函数"""
    engine = QuantFeatureEngine()
    return engine.calculate_technical_features(df)

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    data = {
        'OpenPrice': np.random.uniform(150, 180, 252),
        'HighPrice': np.random.uniform(160, 190, 252),
        'LowPrice': np.random.uniform(140, 170, 252),
        'ClosePrice': np.random.uniform(145, 175, 252),
        'TurnoverVolume': np.random.randint(1000000, 5000000, 252),
        'TurnoverValue': np.random.uniform(100000000, 500000000, 252)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # 计算所有特征
    feature_engine = QuantFeatureEngine()
    feature_df = feature_engine.calculate_all_features(df)
    
    print("特征计算完成！")
    print(f"原始数据列数: {len(df.columns)}")
    print(f"特征数据列数: {len(feature_df.columns)}")
    print(f"新增特征数: {len(feature_df.columns) - len(df.columns)}")
    
    # 显示部分特征
    feature_columns = [col for col in feature_df.columns if col not in df.columns]
    print(f"\n新增特征列: {feature_columns[:10]}...")
