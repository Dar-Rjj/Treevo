import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Abnormal High-Low Range Persistence
    daily_range = df['high'] - df['low']
    range_5d_avg = daily_range.rolling(window=5, min_periods=5).mean()
    range_20d_avg = daily_range.rolling(window=20, min_periods=20).mean()
    
    abnormal_condition = range_5d_avg > range_20d_avg
    abnormal_days = abnormal_condition.rolling(window=10, min_periods=1).apply(
        lambda x: len([i for i in range(len(x)) if all(x[max(0, i-4):i+1])]), raw=False
    )
    one_day_return = df['close'].pct_change()
    signal1 = abnormal_days * daily_range * np.sign(one_day_return)
    
    # Volume-Weighted Price Acceleration
    return_5d = df['close'].pct_change(5)
    return_10d = df['close'].pct_change(10)
    acceleration = return_5d - return_10d
    volume_5d_avg = df['volume'].rolling(window=5, min_periods=5).mean()
    volume_20d_avg = df['volume'].rolling(window=20, min_periods=20).mean()
    volume_trend = volume_5d_avg / volume_20d_avg
    signal2 = np.sign(acceleration) * np.abs(acceleration) * volume_trend
    
    # Relative Strength Divergence
    stock_10d_return = df['close'].pct_change(10)
    market_10d_return = df['close'].pct_change(10)  # Using same stock as market proxy
    strength_diff = stock_10d_return - market_10d_return
    strength_diff_change = strength_diff.diff()
    divergence = strength_diff_change.rolling(window=5, min_periods=1).apply(
        lambda x: sum(x > 0), raw=False
    )
    volume_20d_avg_2 = df['volume'].rolling(window=20, min_periods=20).mean()
    signal3 = divergence * strength_diff * (df['volume'] / volume_20d_avg_2)
    
    # Volatility Regime Breakout
    volatility_5d = df['close'].pct_change().rolling(window=5, min_periods=5).std()
    volatility_20d = df['close'].pct_change().rolling(window=20, min_periods=20).std()
    regime = volatility_5d / volatility_20d
    range_5d_avg_2 = daily_range.rolling(window=5, min_periods=5).mean()
    breakout = (daily_range > range_5d_avg_2) & (df['volume'] > volume_20d_avg)
    signal4 = breakout.astype(float) * (daily_range / range_5d_avg_2) * regime
    
    # Liquidity-Adjusted Momentum Reversal
    one_day_return_2 = df['close'].pct_change()
    five_day_cum_return = (1 + df['close'].pct_change()).rolling(window=5, min_periods=5).apply(
        lambda x: np.prod(1 + x) - 1, raw=False
    )
    reversal = one_day_return_2 * five_day_cum_return
    liquidity_score = df['volume'] / (daily_range + 1e-8)
    vol_diff_sign = np.sign(volatility_5d - volatility_20d)
    signal5 = reversal * liquidity_score * vol_diff_sign
    
    # Multi-timeframe Order Flow Imbalance
    net_buying_volume = df['amount'] / (df['close'] + 1e-8)
    flow_5d = net_buying_volume.rolling(window=5, min_periods=5).sum()
    flow_20d = net_buying_volume.rolling(window=20, min_periods=20).sum()
    flow_ratio = flow_5d / (flow_20d + 1e-8)
    acceleration_flow = flow_ratio - flow_ratio.shift(1)
    signal6 = acceleration_flow * net_buying_volume * np.sign(one_day_return)
    
    # Combine signals with equal weights
    combined_signal = (
        signal1.fillna(0) + signal2.fillna(0) + signal3.fillna(0) + 
        signal4.fillna(0) + signal5.fillna(0) + signal6.fillna(0)
    )
    
    return combined_signal
