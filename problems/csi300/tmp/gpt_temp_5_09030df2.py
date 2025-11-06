import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Momentum Efficiency
    close = df['close']
    
    # Calculate returns
    ret_5d = close.pct_change(5)
    ret_10d = close.pct_change(10)
    
    # Compute momentum divergence
    momentum_divergence = ret_5d - ret_10d
    
    # Calculate return autocorrelation at lag 1 as inefficiency measure
    daily_returns = close.pct_change()
    autocorr = daily_returns.rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 5 else np.nan, raw=False
    )
    
    # Assess Volume-Liquidity Context
    volume = df['volume']
    amount = df['amount']
    
    # Calculate volume averages
    vol_5d_avg = volume.rolling(window=5, min_periods=3).mean()
    vol_10d_avg = volume.rolling(window=10, min_periods=5).mean()
    
    # Compute volume ratio
    volume_ratio = vol_5d_avg / vol_10d_avg
    
    # Calculate volume concentration
    volume_concentration = volume / vol_10d_avg
    
    # Generate Adaptive Alpha Signal
    # Multiply momentum divergence by volume ratio
    signal = momentum_divergence * volume_ratio
    
    # Adjust by price inefficiency (multiply by autocorrelation)
    signal = signal * autocorr
    
    # Scale by liquidity conditions (multiply by volume concentration)
    signal = signal * volume_concentration
    
    # Apply sign from short-term momentum (5-day return direction)
    signal = signal * np.sign(ret_5d)
    
    return signal
