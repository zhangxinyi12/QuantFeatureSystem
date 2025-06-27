#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测分析模块
提供全面的策略评估指标、风险分析和性能报告
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class BacktestAnalyzer:
    """回测分析器"""
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化回测分析器
        
        Args:
            risk_free_rate: 无风险利率
        """
        self.risk_free_rate = risk_free_rate
        self.results = {}
        
    def calculate_performance_metrics(self, 
                                    returns: pd.Series,
                                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        计算性能指标
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            性能指标字典
        """
        # 移除缺失值
        returns = returns.dropna()
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.dropna()
        
        # 基础指标
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 风险调整收益指标
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # 卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # 胜率和盈亏比
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 最大连续亏损
        consecutive_losses = self._calculate_consecutive_losses(returns)
        max_consecutive_losses = consecutive_losses.max() if len(consecutive_losses) > 0 else 0
        
        # 信息比率（如果有基准）
        information_ratio = 0
        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # 贝塔系数
        beta = 1.0
        if benchmark_returns is not None:
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # 阿尔法系数
        alpha = annual_return - (self.risk_free_rate + beta * (benchmark_returns.mean() * 252 - self.risk_free_rate)) if benchmark_returns is not None else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_consecutive_losses': max_consecutive_losses,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha
        }
        
        return metrics
    
    def _calculate_consecutive_losses(self, returns: pd.Series) -> pd.Series:
        """计算连续亏损次数"""
        consecutive_losses = []
        current_losses = 0
        
        for ret in returns:
            if ret < 0:
                current_losses += 1
            else:
                if current_losses > 0:
                    consecutive_losses.append(current_losses)
                current_losses = 0
        
        if current_losses > 0:
            consecutive_losses.append(current_losses)
        
        return pd.Series(consecutive_losses)
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """计算风险指标"""
        returns = returns.dropna()
        
        # VaR (Value at Risk)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # CVaR (Conditional Value at Risk)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # 偏度和峰度
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # 尾部风险
        tail_risk = returns[returns < returns.quantile(0.01)].std()
        
        # 波动率聚集
        volatility_clustering = returns.rolling(20).std().autocorr()
        
        risk_metrics = {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_risk': tail_risk,
            'volatility_clustering': volatility_clustering
        }
        
        return risk_metrics
    
    def calculate_turnover_metrics(self, 
                                 positions: pd.Series,
                                 prices: pd.Series) -> Dict[str, float]:
        """计算换手率指标"""
        if positions is None or prices is None:
            return {}
        
        # 计算持仓变化
        position_changes = positions.diff().abs()
        
        # 换手率
        turnover_rate = position_changes.sum() / len(positions)
        
        # 平均持仓时间
        holding_periods = self._calculate_holding_periods(positions)
        avg_holding_period = holding_periods.mean() if len(holding_periods) > 0 else 0
        
        # 交易成本估算（假设0.1%的交易成本）
        transaction_cost = 0.001
        total_cost = position_changes.sum() * transaction_cost
        
        turnover_metrics = {
            'turnover_rate': turnover_rate,
            'avg_holding_period': avg_holding_period,
            'total_transaction_cost': total_cost
        }
        
        return turnover_metrics
    
    def _calculate_holding_periods(self, positions: pd.Series) -> pd.Series:
        """计算持仓时间"""
        holding_periods = []
        current_period = 0
        
        for pos in positions:
            if pos != 0:
                current_period += 1
            else:
                if current_period > 0:
                    holding_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            holding_periods.append(current_period)
        
        return pd.Series(holding_periods)
    
    def analyze_drawdowns(self, returns: pd.Series) -> Dict[str, List]:
        """分析回撤"""
        returns = returns.dropna()
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        # 找到回撤期间
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append({
                    'start_date': returns.index[start_idx],
                    'end_date': returns.index[i-1],
                    'duration': i - start_idx,
                    'max_drawdown': drawdown.iloc[start_idx:i].min(),
                    'recovery_time': i - start_idx
                })
        
        # 如果仍在回撤中
        if in_drawdown:
            drawdown_periods.append({
                'start_date': returns.index[start_idx],
                'end_date': returns.index[-1],
                'duration': len(returns) - start_idx,
                'max_drawdown': drawdown.iloc[start_idx:].min(),
                'recovery_time': None
            })
        
        return {
            'drawdown_periods': drawdown_periods,
            'total_drawdowns': len(drawdown_periods),
            'avg_drawdown_duration': np.mean([d['duration'] for d in drawdown_periods]) if drawdown_periods else 0,
            'avg_recovery_time': np.mean([d['recovery_time'] for d in drawdown_periods if d['recovery_time'] is not None]) if drawdown_periods else 0
        }
    
    def calculate_rolling_metrics(self, 
                                returns: pd.Series,
                                window: int = 252) -> pd.DataFrame:
        """计算滚动指标"""
        returns = returns.dropna()
        
        rolling_metrics = pd.DataFrame()
        
        # 滚动收益率
        rolling_metrics['rolling_return'] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # 滚动波动率
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # 滚动夏普比率
        rolling_metrics['rolling_sharpe'] = (
            (rolling_metrics['rolling_return'] * 252 / window - self.risk_free_rate) / 
            rolling_metrics['rolling_volatility']
        )
        
        # 滚动最大回撤
        rolling_metrics['rolling_max_drawdown'] = returns.rolling(window).apply(
            lambda x: self._calculate_rolling_drawdown(x)
        )
        
        return rolling_metrics
    
    def _calculate_rolling_drawdown(self, returns: pd.Series) -> float:
        """计算滚动最大回撤"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def generate_performance_report(self, 
                                  returns: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  positions: Optional[pd.Series] = None,
                                  prices: Optional[pd.Series] = None) -> Dict[str, Dict]:
        """生成完整的性能报告"""
        
        # 计算各类指标
        performance_metrics = self.calculate_performance_metrics(returns, benchmark_returns)
        risk_metrics = self.calculate_risk_metrics(returns)
        drawdown_analysis = self.analyze_drawdowns(returns)
        
        # 换手率指标（如果有持仓数据）
        turnover_metrics = {}
        if positions is not None and prices is not None:
            turnover_metrics = self.calculate_turnover_metrics(positions, prices)
        
        # 滚动指标
        rolling_metrics = self.calculate_rolling_metrics(returns)
        
        # 汇总报告
        report = {
            'performance': performance_metrics,
            'risk': risk_metrics,
            'drawdowns': drawdown_analysis,
            'turnover': turnover_metrics,
            'rolling': rolling_metrics
        }
        
        self.results = report
        return report
    
    def plot_performance_analysis(self, 
                                returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                save_path: Optional[str] = None):
        """绘制性能分析图表"""
        
        if not self.results:
            self.generate_performance_report(returns, benchmark_returns)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('策略性能分析', fontsize=16)
        
        # 1. 累积收益曲线
        cumulative_returns = (1 + returns).cumprod()
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, label='策略收益', linewidth=2)
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                           label='基准收益', linewidth=2, alpha=0.7)
        
        axes[0, 0].set_title('累积收益曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 回撤曲线
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[0, 1].set_title('回撤曲线')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 滚动夏普比率
        if 'rolling' in self.results:
            rolling_sharpe = self.results['rolling']['rolling_sharpe']
            axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=2)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('滚动夏普比率')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 收益率分布
        axes[1, 1].hist(returns.values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].axvline(x=returns.mean(), color='red', linestyle='--', label=f'均值: {returns.mean():.4f}')
        axes[1, 1].set_title('收益率分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_summary_report(self):
        """打印汇总报告"""
        if not self.results:
            logger.warning("没有可用的分析结果")
            return
        
        print("=" * 60)
        print("策略性能汇总报告")
        print("=" * 60)
        
        # 性能指标
        perf = self.results['performance']
        print(f"\n📈 收益指标:")
        print(f"  总收益率: {perf['total_return']:.2%}")
        print(f"  年化收益率: {perf['annual_return']:.2%}")
        print(f"  年化波动率: {perf['volatility']:.2%}")
        print(f"  夏普比率: {perf['sharpe_ratio']:.2f}")
        print(f"  索提诺比率: {perf['sortino_ratio']:.2f}")
        print(f"  卡玛比率: {perf['calmar_ratio']:.2f}")
        
        # 风险指标
        risk = self.results['risk']
        print(f"\n⚠️ 风险指标:")
        print(f"  最大回撤: {perf['max_drawdown']:.2%}")
        print(f"  95% VaR: {risk['var_95']:.2%}")
        print(f"  99% VaR: {risk['var_99']:.2%}")
        print(f"  偏度: {risk['skewness']:.2f}")
        print(f"  峰度: {risk['kurtosis']:.2f}")
        
        # 交易指标
        print(f"\n📊 交易指标:")
        print(f"  胜率: {perf['win_rate']:.2%}")
        print(f"  盈亏比: {perf['profit_factor']:.2f}")
        print(f"  平均盈利: {perf['avg_win']:.2%}")
        print(f"  平均亏损: {perf['avg_loss']:.2%}")
        print(f"  最大连续亏损: {perf['max_consecutive_losses']:.0f}")
        
        # 回撤分析
        dd = self.results['drawdowns']
        print(f"\n📉 回撤分析:")
        print(f"  总回撤次数: {dd['total_drawdowns']}")
        print(f"  平均回撤持续时间: {dd['avg_drawdown_duration']:.0f} 天")
        print(f"  平均恢复时间: {dd['avg_recovery_time']:.0f} 天")
        
        print("=" * 60) 