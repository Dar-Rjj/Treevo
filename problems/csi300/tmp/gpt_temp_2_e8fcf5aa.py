import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Convergence
    data['momentum_3d'] = data['close'].pct_change(3)
    data['momentum_10d'] = data['close'].pct_change(10)
    
    # Direction alignment multiplier
    data['momentum_alignment'] = np.where(
        data['momentum_3d'] * data['momentum_10d'] > 0, 
        1.5,  # Strong alignment
        np.where(data['momentum_3d'] * data['momentum_10d'] < 0, 0.5, 1.0)  # Divergence or neutral
    )
    
    # Volatility Regime Identification
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # High-Low range percentiles
    data['hl_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_percentile'] = data['hl_range'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)) * 1 + (x.iloc[-1] > np.percentile(x, 90)) * 1,
        raw=False
    )
    
    # Regime-Adjusted Momentum
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    data['regime_momentum'] = (
        (data['momentum_3d'] + data['momentum_10d']) / 2 * 
        data['momentum_alignment'] * 
        (1 + data['volatility_percentile'] * 0.2) *
        np.sign(data['intraday_momentum'])
    )
    
    # Volume-Expansion Breakout
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    data['close_breakthrough'] = np.where(
        data['close'] > data['high'].rolling(window=5, min_periods=3).max().shift(1), 1,
        np.where(data['close'] < data['low'].rolling(window=5, min_periods=3).min().shift(1), -1, 0)
    )
    
    # Price Range Efficiency
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_sum'] = data['daily_efficiency'].rolling(window=6, min_periods=3).sum()
    
    # Volume-Weighted Momentum
    data['vw_efficiency'] = data['daily_efficiency'] * data['volume_ratio']
    data['intraday_strength'] = (data['close'] - (data['high'] + data['low']) / 2) / ((data['high'] - data['low']) / 2).replace(0, np.nan)
    data['volume_weighted_momentum'] = data['vw_efficiency'] * (1 + data['intraday_strength'])
    
    # Order Flow Multiplier
    data['avg_trade_size'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['trade_size_deviation'] = (
        data['avg_trade_size'] / 
        data['avg_trade_size'].rolling(window=20, min_periods=10).median() - 1
    )
    data['order_flow_multiplier'] = 1 + np.tanh(data['trade_size_deviation'] * 2)
    
    # Volatility-Weighted Synthesis
    volatility_weight = 1 / (data['hl_range'].rolling(window=10, min_periods=5).mean() + 0.001)
    
    factor = (
        data['regime_momentum'] * 
        data['volume_weighted_momentum'] * 
        volatility_weight /
        (abs(data['efficiency_sum']) + 0.001) *
        data['order_flow_multiplier'] *
        data['close_breakthrough']
    )
    
    return factor.replace([np.inf, -np.inf], np.nan).fillna(0)
