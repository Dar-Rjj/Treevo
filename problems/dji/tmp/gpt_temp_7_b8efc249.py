import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume Impact Score
    sum_volume_10d = df['volume'].rolling(window=10).sum()
    avg_high_10d = df['high'].rolling(window=10).mean()
    avg_low_10d = df['low'].rolling(window=10).mean()
    volume_impact_score = (avg_high_10d - avg_low_10d) / sum_volume_10d
    
    # Generate Factor 1
    factor_1 = intraday_return * volume_impact_score
    
    # Calculate Price Momentum
    recent_close = df['close']
    close_10d_ago = df['close'].shift(10)
    price_momentum = recent_close - close_10d_ago
    
    # Adjust by Volume
    cumulative_volume = df['volume'].rolling(window=10).sum()
    adjusted_momentum = price_momentum / cumulative_volume
    
    # Calculate Cumulative Volume-Weighted Momentum
    cumulative_volume_weighted_momentum = (adjusted_momentum * df['volume']).rolling(window=10).sum()
    
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Opening Price Trend
    opening_price_trend = df['open'] - df['close'].shift(1)
    
    # Calculate Volume-Weighted Intraday Movement (VWIM)
    vwim = intraday_high_low_spread * df['volume']
    vwim_10d_ma = vwim.rolling(window=10).mean()
    
    # Calculate Amount-Weighted Opening Trend (AWOT)
    awot = opening_price_trend * df['amount']
    awot_10d_ma = awot.rolling(window=10).mean()
    
    # Combine Adjusted Movements and Trends (AMT)
    amt = vwim + awot - vwim_10d_ma - awot_10d_ma
    
    # Adjust by Cumulative Volume (CV)
    cv = df['volume'].rolling(window=10).sum()
    amt_adjusted = amt / cv
    
    # Confirm with Enhanced Volume Trend
    avg_volume_20d = df['volume'].rolling(window=20).mean()
    current_day_volume = df['volume']
    volume_ratio = current_day_volume / avg_volume_20d
    
    if volume_ratio > 1.6:
        combined_momentum = price_momentum * intraday_return
        aggregated_momentum = cumulative_volume_weighted_momentum.sum()
        final_factor = factor_1 * aggregated_momentum
    else:
        final_factor = 0.4 * (price_momentum + intraday_return)
    
    return final_factor
