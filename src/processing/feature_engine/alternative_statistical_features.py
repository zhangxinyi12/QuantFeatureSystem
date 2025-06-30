import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import pywt

class AlternativeFeatures:
    def __init__(self, df):
        """初始化另类数据特征计算器"""
        self.df = df.copy()
    
    def calculate_sentiment_scores(self, text_col='news_text'):
        """计算新闻/社交媒体情感分析得分"""
        if text_col not in self.df.columns:
            raise ValueError(f"文本列 '{text_col}' 不存在")
        
        # 情感分析
        self.df['sentiment'] = self.df[text_col].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity)
        
        # 关键词频率
        keywords = ['earnings', 'growth', 'acquisition', 'merger', 'lawsuit', 'regulation']
        vectorizer = CountVectorizer(vocabulary=keywords)
        keyword_counts = vectorizer.fit_transform(self.df[text_col].fillna(''))
        keyword_df = pd.DataFrame(keyword_counts.toarray(), columns=keywords)
        
        # 合并关键词特征
        self.df = pd.concat([self.df, keyword_df.add_prefix('keyword_')], axis=1)
        
        # 事件类型识别
        event_types = {
            r'earnings report|quarterly results': 'earnings',
            r'acquisition|merger|takeover': 'm&a',
            r'layoff|job cut|restructuring': 'restructuring',
            r'product launch|new product': 'product_launch',
            r'fda approval|regulatory approval': 'regulatory'
        }
        
        self.df['event_type'] = 'other'
        for pattern, event in event_types.items():
            mask = self.df[text_col].str.contains(pattern, case=False, na=False)
            self.df.loc[mask, 'event_type'] = event
        
        return self.df
    
    def calculate_supply_chain_features(self, port_activity_col, shipping_data_col):
        """计算供应链/物流特征"""
        # 港口活动特征
        self.df['port_activity_ma7'] = self.df[port_activity_col].rolling(7).mean()
        self.df['port_activity_change'] = self.df[port_activity_col].pct_change()
        
        # 航运数据特征
        self.df['shipping_volume_ma7'] = self.df[shipping_data_col].rolling(7).mean()
        self.df['shipping_volume_change'] = self.df[shipping_data_col].pct_change()
        
        # 活动与航运相关性
        self.df['port_ship_correlation'] = self.df[['port_activity_col', shipping_data_col]].rolling(30).corr().iloc[0::2, 1].values
        
        return self.df
    
    def calculate_consumer_behavior(self, search_volume_col, app_downloads_col):
        """计算消费者行为特征"""
        # 搜索量特征
        self.df['search_volume_ma7'] = self.df[search_volume_col].rolling(7).mean()
        self.df['search_volume_change'] = self.df[search_volume_col].pct_change()
        
        # APP下载特征
        self.df['app_downloads_ma7'] = self.df[app_downloads_col].rolling(7).mean()
        self.df['app_downloads_change'] = self.df[app_downloads_col].pct_change()
        
        # 搜索与下载相关性
        self.df['search_download_correlation'] = self.df[[search_volume_col, app_downloads_col]].rolling(30).corr().iloc[0::2, 1].values
        
        return self.df
    
    def calculate_weather_impact(self, weather_data_col, commodity_col):
        """计算天气对商品的影响"""
        # 天气异常指标
        self.df['weather_anomaly'] = (self.df[weather_data_col] - 
                                     self.df[weather_data_col].rolling(30).mean()) / 
                                     self.df[weather_data_col].rolling(30).std()
        
        # 天气与商品价格相关性
        self.df['weather_commodity_corr'] = self.df[[weather_data_col, commodity_col]].rolling(30).corr().iloc[0::2, 1].values
        
        # 极端天气事件
        self.df['extreme_weather'] = np.where(np.abs(self.df['weather_anomaly']) > 2, 1, 0)
        
        return self.df


class StatisticalFeatures:
    def __init__(self, df):
        """初始化统计变换特征计算器"""
        self.df = df.copy()
    
    def apply_normalization(self, cols, method='zscore'):
        """应用标准化/归一化"""
        for col in cols:
            if method == 'zscore':
                scaler = StandardScaler()
                self.df[f'{col}_zscore'] = scaler.fit_transform(self.df[[col]])
            elif method == 'minmax':
                scaler = MinMaxScaler()
                self.df[f'{col}_minmax'] = scaler.fit_transform(self.df[[col]])
            else:
                raise ValueError("不支持的标准化方法，请选择 'zscore' 或 'minmax'")
        return self.df
    
    def apply_differencing(self, cols, order=1):
        """应用差分"""
        for col in cols:
            self.df[f'{col}_diff{order}'] = self.df[col].diff(order)
        return self.df
    
    def apply_log_transform(self, cols):
        """应用对数变换"""
        for col in cols:
            # 添加小常数避免零值问题
            self.df[f'{col}_log'] = np.log(self.df[col] + 1e-8)
        return self.df
    
    def apply_binning(self, cols, bins=5, method='quantile'):
        """应用分箱"""
        for col in cols:
            if method == 'quantile':
                self.df[f'{col}_bin'] = pd.qcut(self.df[col], bins, labels=False, duplicates='drop')
            elif method == 'equal':
                self.df[f'{col}_bin'] = pd.cut(self.df[col], bins, labels=False, duplicates='drop')
            else:
                raise ValueError("不支持的分箱方法，请选择 'quantile' 或 'equal'")
        return self.df
    
    def create_interaction_terms(self, col_pairs):
        """创建交互项"""
        for col1, col2 in col_pairs:
            # 比值交互项
            self.df[f'{col1}_div_{col2}'] = self.df[col1] / (self.df[col2] + 1e-8)
            
            # 乘积交互项
            self.df[f'{col1}_mul_{col2}'] = self.df[col1] * self.df[col2]
            
            # 差值交互项
            self.df[f'{col1}_sub_{col2}'] = self.df[col1] - self.df[col2]
        return self.df
    
    def apply_pca(self, cols, n_components=3, prefix='pca'):
        """应用主成分分析"""
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[cols])
        
        # 应用PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(scaled_data)
        
        # 添加PCA特征
        for i in range(n_components):
            self.df[f'{prefix}_{i+1}'] = pca_features[:, i]
        
        return self.df, pca
    
    def apply_fourier_transform(self, col, n_components=3):
        """应用傅里叶变换"""
        # 获取时间序列数据
        series = self.df[col].values
        
        # 应用FFT
        fft_result = np.fft.fft(series)
        frequencies = np.fft.fftfreq(len(series))
        
        # 提取主要频率分量
        idx = np.argsort(np.abs(fft_result))[::-1]
        top_frequencies = frequencies[idx[:n_components]]
        top_amplitudes = np.abs(fft_result)[idx[:n_components]]
        
        # 添加特征
        for i in range(n_components):
            self.df[f'{col}_freq_{i+1}'] = top_frequencies[i]
            self.df[f'{col}_amp_{i+1}'] = top_amplitudes[i]
        
        return self.df
    
    def apply_wavelet_transform(self, col, wavelet='db4', level=3):
        """应用小波变换"""
        # 获取时间序列数据
        series = self.df[col].values
        
        # 应用小波变换
        coeffs = pywt.wavedec(series, wavelet, level=level)
        
        # 添加特征
        for i, coeff in enumerate(coeffs):
            # 近似系数
            if i == 0:
                self.df[f'{col}_wavelet_approx'] = np.mean(coeff)
            # 细节系数
            else:
                self.df[f'{col}_wavelet_detail_{i}'] = np.mean(coeff)
        
        return self.df


if __name__ == "__main__":
    # 创建模拟另类数据
    dates = pd.date_range(start='2023-01-01', periods=100)
    data = {
        'date': dates,
        'news_text': [f"Company reported strong earnings growth in Q{np.random.randint(1,5)}" 
                      for _ in range(100)],
        'port_activity': np.random.randint(50, 200, 100),
        'shipping_volume': np.random.randint(100, 500, 100),
        'search_volume': np.random.randint(1000, 5000, 100),
        'app_downloads': np.random.randint(500, 2000, 100),
        'temperature': np.random.uniform(10, 35, 100),
        'commodity_price': np.cumsum(np.random.normal(0, 0.5, 100)) + 50,
        'stock_price': np.cumsum(np.random.normal(0, 1, 100)) + 100,
        'trading_volume': np.random.randint(1000000, 5000000, 100)
    }
    alt_df = pd.DataFrame(data)
    
    # 计算另类数据特征
    af = AlternativeFeatures(alt_df)
    alt_features = af.calculate_sentiment_scores('news_text')
    alt_features = af.calculate_supply_chain_features('port_activity', 'shipping_volume')
    alt_features = af.calculate_consumer_behavior('search_volume', 'app_downloads')
    alt_features = af.calculate_weather_impact('temperature', 'commodity_price')
    
    # 应用统计变换
    sf = StatisticalFeatures(alt_features)
    
    # 标准化
    sf.apply_normalization(['stock_price', 'trading_volume'], method='zscore')
    
    # 差分
    sf.apply_differencing(['stock_price', 'commodity_price'], order=1)
    
    # 对数变换
    sf.apply_log_transform(['trading_volume', 'search_volume'])
    
    # 分箱
    sf.apply_binning(['sentiment'], bins=5, method='quantile')
    
    # 交互项
    sf.create_interaction_terms([('stock_price', 'trading_volume'), 
                               ('commodity_price', 'temperature')])
    
    # PCA
    pca_cols = ['stock_price', 'trading_volume', 'commodity_price', 'sentiment']
    sf.apply_pca(pca_cols, n_components=2, prefix='market_pca')
    
    # 傅里叶变换
    sf.apply_fourier_transform('stock_price', n_components=3)
    
    # 小波变换
    sf.apply_wavelet_transform('commodity_price', wavelet='db4', level=3)
    
    # 获取最终DataFrame
    final_df = sf.df
    
    # 保存结果
    final_df.to_csv('alternative_statistical_features.csv', index=False)
    
    # 打印结果
    print("另类数据和统计变换特征计算完成!")
    print("特征列:", final_df.columns.tolist())
    print("示例数据:")
    print(final_df.head())
    
    # 可视化傅里叶变换结果
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # 原始价格序列
    plt.subplot(2, 2, 1)
    plt.plot(final_df['date'], final_df['stock_price'])
    plt.title('股票价格')
    
    # FFT振幅
    plt.subplot(2, 2, 2)
    plt.bar(range(1, 4), final_df[['stock_price_amp_1', 'stock_price_amp_2', 'stock_price_amp_3']].iloc[0])
    plt.title('主要频率振幅')
    plt.xlabel('频率分量')
    
    # PCA结果
    plt.subplot(2, 2, 3)
    plt.scatter(final_df['market_pca_1'], final_df['market_pca_2'])
    plt.title('PCA结果')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    
    # 情感分数分布
    plt.subplot(2, 2, 4)
    plt.hist(final_df['sentiment'], bins=20)
    plt.title('情感分数分布')
    
    plt.tight_layout()
    plt.savefig('alternative_statistical_features_visualization.png')
    plt.show()