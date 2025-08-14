import pandas as pd
import pandas as pd

def heuristics_v2(df, n_days=10, m_days=5):
    # Calculate Intraday Volume Proportion
    intraday_volume_proportion = df['volume'] - (df['volume'].shift(1) + df['volume'].shift(2))
    
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Calculate Intraday Volume Intensity
    high_low_range = df['high'] - df['low']
    intraday_volume_intensity = intraday_volume_proportion / high_low_range
    
    # Calculate Volume Impact Score
    volume_sum_n_days = df['volume'].rolling(window=n_days).sum()
    avg_high_n_days = df['high'].rolling(window=n_days).mean()
    avg_low_n_days = df['low'].rolling(window=n_days).mean()
    volume_impact_score = (avg_high_n_days - avg_low_n_days) / volume_sum_n_days
    
    # Calculate Weighted Intraday Volume Intensity
    weighted_intraday_volume_intensity = intraday_volume_intensity * intraday_return
    
    # Calculate Intraday and Breakout Momentum
    high_low_spread = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Adjust by Cumulative Volume
    cumulative_volume = df['volume'].rolling(window=n_days).sum()
    adjusted_momentum = high_low_spread / cumulative_volume
    
    # Generate Factor 1
    factor_1 = intraday_return * volume_impact_score
    
    # Confirm with Volume Trend
    avg_volume_5_days = df['volume'].rolling(window=5).mean()
    current_day_volume = df['volume']
    volume_ratio = current_day_volume / avg_volume_5_days
    factor_2 = volume_ratio.apply(lambda x: (intraday_return * high_low_spread) if x > 1.2 else 0)
    
    # Adjust by Enhanced Volume Trend
    avg_volume_15_days = df['volume'].rolling(window=15).mean()
    enhanced_volume_ratio = current_day_volume / avg_volume_15_days
    if enhanced_volume_ratio > 1.8:
        momentum_aggregate = (intraday_return * high_low_spread).rolling(window=m_days).sum()
    else:
        momentum_aggregate = 0.4 * (high_low_spread + intraday_return)
    
    # Final Factor
    final_factor = factor_1 * momentum_aggregate
    
    return final_factor
