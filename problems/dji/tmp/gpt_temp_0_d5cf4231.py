import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute High-Low Delta and Close-Open Delta
    high_low_delta = df['high'] - df['low']
    close_open_delta = df['close'] - df['open']
    
    # Combine Deltas into Momentum Score
    momentum_score = high_low_delta + close_open_delta
    momentum_score = momentum_score.ewm(alpha=0.1, adjust=False).mean()  # Exponential decay
    
    # Calculate Volume Change
    volume_change = df['volume'] - df['volume'].shift(1)
    
    # Apply Volume Weighting
    min_volume_increase = 0.1 * df['volume'].shift(1)  # 10% increase threshold
    min_momentum_threshold = 0.05  # Minimum momentum score threshold
    significant_signals = (volume_change > min_volume_increase) & (momentum_score > min_momentum_threshold)
    
    # Integrate Price and Volume Changes
    price_change = df['close'] - df['close'].shift(1)
    integrated_changes = price_change * volume_change
    integrated_changes = integrated_changes.where(significant_signals, 0)
    
    # Cumulative Momentum Indicator
    cumulative_momentum = integrated_changes.rolling(window=20).sum()
    
    # Smooth the Indicator
    daily_volatility = df['close'].pct_change().rolling(window=20).std()
    optimal_window = (1 / daily_volatility).astype(int).clip(lower=5, upper=60)  # Adjust window size based on volatility
    smoothed_indicator = cumulative_momentum.rolling(window=optimal_window, min_periods=1).mean()
    
    return smoothed_indicator
