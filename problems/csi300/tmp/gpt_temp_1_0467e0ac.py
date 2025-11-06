import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Factor
    # Price Momentum Component
    price_5d = df['close'] / df['close'].shift(5) - 1
    price_20d = df['close'] / df['close'].shift(20) - 1
    
    # Volume Momentum Component
    volume_5d = df['volume'] / df['volume'].shift(5) - 1
    volume_20d = df['volume'] / df['volume'].shift(20) - 1
    
    # Divergence Signal
    short_divergence = price_5d - volume_5d
    medium_divergence = price_20d - volume_20d
    
    # Range Efficiency Factor
    # Daily Range Calculation
    daily_range = (df['high'] - df['low']) / df['close']
    avg_range_10d = daily_range.rolling(window=10).mean()
    
    # Price Movement
    abs_return = abs(df['close'] / df['close'].shift(1) - 1)
    cum_movement_5d = abs_return.rolling(window=5).sum()
    
    # Efficiency Score
    daily_efficiency = abs_return / daily_range
    sum_ranges_5d = daily_range.rolling(window=5).sum()
    efficiency_5d = cum_movement_5d / sum_ranges_5d
    
    # Volume-Confirmed Reversal Factor
    # Price Extreme Detection
    min_5d = df['low'].rolling(window=5).min()
    max_5d = df['high'].rolling(window=5).max()
    extreme_position = (df['close'] - min_5d) / (max_5d - min_5d)
    
    daily_return = df['close'] / df['close'].shift(1) - 1
    return_mean_20d = daily_return.rolling(window=20).mean()
    return_std_20d = daily_return.rolling(window=20).std()
    return_zscore = (daily_return - return_mean_20d) / return_std_20d
    
    # Volume Confirmation
    avg_volume_20d = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / avg_volume_20d
    
    reversal_signal = -np.sign(daily_return) * volume_ratio * extreme_position
    
    # Order Flow Momentum Factor
    # Directional Amount Analysis
    up_days = df['close'] > df['close'].shift(1)
    down_days = df['close'] < df['close'].shift(1)
    
    up_amount = df['amount'].where(up_days, 0)
    down_amount = df['amount'].where(down_days, 0)
    total_amount = df['amount']
    
    net_flow = (up_amount - down_amount) / total_amount
    
    # Flow Persistence
    cum_net_flow_5d = net_flow.rolling(window=5).sum()
    
    def count_same_sign(series):
        return (series > 0).rolling(window=5).sum() if series.iloc[-1] > 0 else (series < 0).rolling(window=5).sum()
    
    direction_consistency = net_flow.rolling(window=5).apply(lambda x: (x > 0).sum() if x.iloc[-1] > 0 else (x < 0).sum(), raw=False)
    flow_momentum = cum_net_flow_5d * direction_consistency
    
    # Combine factors with equal weights
    alpha_factor = (
        0.25 * short_divergence +
        0.25 * medium_divergence +
        0.25 * efficiency_5d +
        0.25 * reversal_signal +
        0.25 * flow_momentum
    )
    
    return alpha_factor
