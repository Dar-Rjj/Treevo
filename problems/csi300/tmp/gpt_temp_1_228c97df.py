import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Efficiency Momentum
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    efficiency_momentum = intraday_efficiency - intraday_efficiency.shift(3)
    daily_range = df['high'] - df['low']
    range_momentum = daily_range - daily_range.shift(3)
    signal1 = efficiency_momentum * range_momentum
    
    # Volume-Stabilized Gap Reversal
    overnight_gap = df['open'] / df['close'].shift(1) - 1
    gap_fill = (df['close'] - df['open']) / overnight_gap.replace(0, np.nan)
    gap_fill_momentum = gap_fill - gap_fill.shift(10)
    volume_std = df['volume'].rolling(window=10, min_periods=1).std()
    signal2 = gap_fill_momentum / volume_std.replace(0, np.nan)
    
    # Range Compression Breakout
    current_range = df['high'] - df['low']
    avg_range = current_range.rolling(window=5, min_periods=1).mean()
    compression_ratio = current_range / avg_range.replace(0, np.nan)
    volume_momentum = df['volume'] - df['volume'].shift(3)
    signal3 = (1 - compression_ratio) * volume_momentum
    
    # Amount-Weighted Velocity
    velocity_4d = df['close'] - df['close'].shift(4)
    velocity_8d = df['close'] - df['close'].shift(8)
    velocity_convergence = velocity_4d - velocity_8d
    efficiency_ratio = velocity_4d / velocity_8d.replace(0, np.nan)
    avg_amount = df['amount'].rolling(window=5, min_periods=1).mean()
    amount_ratio = df['amount'] / avg_amount.replace(0, np.nan)
    signal4 = velocity_convergence * efficiency_ratio * amount_ratio
    
    # Gap-Volume Persistence
    gap = df['open'] / df['close'].shift(1) - 1
    cumulative_gap_persistence = gap.rolling(window=5, min_periods=1).sum()
    volume_percentile = df['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) == 20 else np.nan
    )
    signal5 = cumulative_gap_persistence * volume_percentile
    
    # Range-Amount Momentum
    range_momentum_7d = daily_range - daily_range.shift(7)
    amount_momentum_4d = df['amount'] - df['amount'].shift(4)
    amount_momentum_8d = df['amount'] - df['amount'].shift(8)
    amount_convergence = amount_momentum_4d - amount_momentum_8d
    signal6 = range_momentum_7d * amount_convergence
    
    # Combine signals with equal weights
    combined_signal = (
        signal1.fillna(0) + signal2.fillna(0) + signal3.fillna(0) + 
        signal4.fillna(0) + signal5.fillna(0) + signal6.fillna(0)
    )
    
    return combined_signal
