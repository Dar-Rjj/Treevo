import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume Impact Score
    sum_volume_10d = df['volume'].rolling(window=10).sum()
    avg_high_10d = df['high'].rolling(window=10).mean()
    avg_low_10d = df['low'].rolling(window=10).mean()
    price_range_10d = avg_high_10d - avg_low_10d
    volume_impact_score = price_range_10d / sum_volume_10d
    
    # Generate Factor 1
    factor_1 = intraday_return * volume_impact_score
    
    # Calculate Price Momentum
    recent_close = df['close']
    close_10d_ago = df['close'].shift(10)
    price_momentum = recent_close - close_10d_ago
    
    # Adjust by Volume
    cum_volume_10d = df['volume'].rolling(window=10).sum()
    adjusted_momentum = price_momentum / cum_volume_10d
    
    # Calculate Cumulative Volume-Weighted Momentum
    adj_momentum_vol = adjusted_momentum * df['volume']
    cumulative_vol_weighted_momentum = adj_momentum_vol.rolling(window=10).sum()
    
    # Confirm with Volume Trend
    avg_volume_5d = df['volume'].rolling(window=5).mean()
    current_volume = df['volume']
    volume_ratio = current_volume / avg_volume_5d
    aggregated_momentum = cumulative_vol_weighted_momentum.rolling(window=10).sum()
    
    final_factor = (factor_1 * aggregated_momentum) if volume_ratio > 1 else 0
    
    # Calculate Intraday High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Calculate Opening Price Trend
    opening_price_trend = df['open'] - df['close'].shift(1)
    
    # Calculate Intraday Volume Intensity
    intraday_volume_proportion = df['volume'] - (df['close'].shift(1) + df['open'])
    intraday_volume_intensity = intraday_volume_proportion / high_low_spread
    
    # Calculate Volume-Weighted Intraday Movement (VWIM)
    vwim = high_low_spread * df['volume']
    avg_vwim_5d = vwim.rolling(window=5).mean()
    
    # Calculate Amount-Weighted Opening Trend (AWOT)
    awot = opening_price_trend * df['amount']
    avg_awot_5d = awot.rolling(window=5).mean()
    
    # Combine Adjusted Movements and Trends (AMT)
    amt = vwim + awot - avg_vwim_5d - avg_awot_5d
    
    # Calculate Cumulative Volume (CV)
    cv = df['volume'].rolling(window=10).sum()
    
    # Adjust by Cumulative Volume (CV)
    adj_amt = amt / cv
    
    # Calculate Intraday and Breakout Momentum
    breakout_momentum = (df['high'] - df['low']) / df['close'].shift(1)
    adj_breakout_momentum = breakout_momentum / cv
    
    # Calculate Cumulative Volume-Weighted Momentum (CVWM)
    cvwm = adj_breakout_momentum * df['volume']
    cumulative_cvwm = cvwm.rolling(window=15).sum()
    
    # Confirm with Volume Trend
    avg_volume_15d = df['volume'].rolling(window=15).mean()
    volume_ratio_15d = current_volume / avg_volume_15d
    if volume_ratio_15d > 1.3:
        final_factor_2 = adj_breakout_momentum * adj_amt * cumulative_cvwm
    else:
        final_factor_2 = 0.6 * (adj_breakout_momentum + adj_amt)
    
    # Final Factor
    final_factor = factor_1 * final_factor_2
    
    return final_factor
