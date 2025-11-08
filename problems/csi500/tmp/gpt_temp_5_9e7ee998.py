import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Relative Momentum with Volume Confirmation
    # Calculate Price Momentum
    momentum_10d = data['close'].pct_change(periods=10)
    momentum_20d = data['close'].pct_change(periods=20)
    
    # Compare Momentum Periods
    relative_momentum = (momentum_10d / momentum_20d) - 1
    
    # Add Volume Confirmation
    volume_mean_20d = data['volume'].rolling(window=20).mean()
    volume_std_20d = data['volume'].rolling(window=20).std()
    volume_zscore = (data['volume'] - volume_mean_20d) / volume_std_20d
    volume_zscore_5d = volume_zscore.rolling(window=5).mean()
    
    momentum_factor = np.tanh(relative_momentum * volume_zscore_5d)
    
    # Volatility Regime Adjusted Return
    # Calculate Recent Returns
    return_5d = data['close'].pct_change(periods=5)
    return_1d = data['close'].pct_change()
    
    # Assess Volatility Environment
    vol_20d = data['close'].pct_change().rolling(window=20).std()
    vol_60d = data['close'].pct_change().rolling(window=60).std()
    vol_ratio = vol_20d / vol_60d
    
    # Adjust Returns by Volatility Regime
    adjusted_5d = return_5d * vol_ratio
    adjusted_1d = return_1d * (1 / vol_ratio)
    volatility_factor = 0.7 * adjusted_1d + 0.3 * adjusted_5d
    
    # Opening Gap Persistence Factor
    # Analyze Opening Gaps
    prev_close = data['close'].shift(1)
    gap_pct = (data['open'] - prev_close) / prev_close
    
    # Assess Gap Persistence
    gap_magnitude_sum = gap_pct.abs().rolling(window=5).sum()
    gap_directional_sum = gap_pct.rolling(window=5).sum()
    persistence_ratio = gap_directional_sum / gap_magnitude_sum
    
    # Combine with Volume Pattern
    volume_trend = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    gap_factor = np.cbrt(persistence_ratio * volume_trend)
    
    # High-Low Range Efficiency
    # Calculate Daily Range
    daily_range = (data['high'] - data['low']) / prev_close
    abs_return = data['close'].pct_change().abs()
    
    # Compare with Price Movement
    efficiency = abs_return / daily_range
    
    # Create Efficiency Signal
    efficiency_mean = efficiency.rolling(window=10).mean()
    efficiency_std = efficiency.rolling(window=20).std()
    efficiency_factor = (efficiency - efficiency_mean) / efficiency_std
    
    # Volume-Weighted Price Level Attraction
    # Identify Key Price Levels
    highest_high = data['high'].rolling(window=20).max()
    lowest_low = data['low'].rolling(window=20).min()
    
    # Measure Current Position
    dist_to_high = (highest_high - data['close']) / highest_high
    dist_to_low = (data['close'] - lowest_low) / data['close']
    position_ratio = dist_to_low / (dist_to_high + dist_to_low)
    
    # Add Volume Weighting
    volume_5d_avg = data['volume'].rolling(window=5).mean()
    volume_ratio = data['volume'] / volume_5d_avg
    attraction_factor = 1 / (1 + np.exp(-position_ratio * volume_ratio))
    
    # Combine all factors with equal weighting
    final_factor = (
        momentum_factor.fillna(0) +
        volatility_factor.fillna(0) +
        gap_factor.fillna(0) +
        efficiency_factor.fillna(0) +
        attraction_factor.fillna(0)
    ) / 5
    
    return final_factor
