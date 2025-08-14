import pandas as pd
import pandas as pd

def heuristics_v2(df, n_days=10, m_days=5):
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Calculate Intraday Volume Proportion
    overnight_volume = df['volume'].shift(1) - df['volume']
    
    # Calculate Intraday Volume Intensity
    intraday_vol_intensity = overnight_volume / (df['high'] - df['low'])
    
    # Calculate Volume Impact Score
    sum_volume_n_days = df['volume'].rolling(window=n_days).sum()
    avg_high_low = (df['high'].rolling(window=n_days).mean() - df['low'].rolling(window=n_days).mean()) / sum_volume_n_days
    volume_impact_score = avg_high_low
    
    # Multiply Intraday Return by Volume Impact Score
    intermediate_factor_1 = intraday_return * volume_impact_score
    
    # Calculate Price Momentum
    price_momentum = df['close'] - df['close'].shift(n_days)
    
    # Adjust Price Momentum by Volume
    sum_volume_n_days = df['volume'].rolling(window=n_days).sum()
    adjusted_momentum = price_momentum / sum_volume_n_days
    
    # Calculate Cumulative Volume-Weighted Momentum
    cum_volume_weighted_momentum = (adjusted_momentum * df['volume']).rolling(window=n_days).sum()
    
    # Confirm with Volume Trend
    avg_volume_5_days = df['volume'].rolling(window=m_days).mean()
    volume_ratio = df['volume'] / avg_volume_5_days
    
    combined_momentum = 0
    if volume_ratio > 1:
        combined_momentum = price_momentum * intraday_return
        aggregated_momentum = combined_momentum.rolling(window=m_days).sum()
        final_factor = intermediate_factor_1 * aggregated_momentum
    else:
        final_factor = 0
    
    # Incorporate Volume-Adjusted Breakout Intensity
    breakout_intensity = (df['high'] - df['low']) / df['close'].shift(1)
    sum_volume_n_days = df['volume'].rolling(window=n_days).sum()
    adjusted_breakout_intensity = breakout_intensity / sum_volume_n_days
    combined_intensities = intraday_vol_intensity + adjusted_breakout_intensity
    weighted_combined_intensities = combined_intensities * intraday_return
    
    return final_factor + weighted_combined_intensities
