import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    # Short-Term Momentum (M1): Close at t vs Close at t-5
    M1 = df['close'] / df['close'].shift(5) - 1
    
    # Medium-Term Momentum (M2): Close at t vs Close at t-20
    M2 = df['close'] / df['close'].shift(20) - 1
    
    # Compute Momentum Divergence
    momentum_divergence = np.log((1 + M1) / (1 + M2))
    
    # Calculate Volatility Adjustment
    # Compute High-Low Range
    high_low_range = (df['high'] - df['low']) / df['close']
    
    # Calculate Average Range over 10 days
    avg_range = high_low_range.rolling(window=10, min_periods=1).mean()
    
    # Calculate Volume Percentile over 20-day window
    volume_percentile = df['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Combine Components
    # Divide Momentum Divergence by Volatility and multiply by Volume Percentile
    factor = momentum_divergence / (avg_range + 1e-8) * volume_percentile
    
    return factor
