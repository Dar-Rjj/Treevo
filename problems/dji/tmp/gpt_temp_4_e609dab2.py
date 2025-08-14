import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Opening Price Trend
    opening_price_trend = df['open'] - df['close'].shift(1)
    
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
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
    
    # Calculate Price Momentum
    price_momentum = df['close'] - df['close'].shift(10)
    
    # Calculate Intraday Momentum
    intraday_momentum = (df['high'] - df['low']) / df['low']
    
    # Confirm with Volume Trend
    av_20d = df['volume'].rolling(window=20).mean()
    vr = df['volume'] / av_20d
    
    # Volume Impact Score
    sum_volume_10d = df['volume'].rolling(window=10).sum()
    avg_high_10d = df['high'].rolling(window=10).mean()
    avg_low_10d = df['low'].rolling(window=10).mean()
    volume_impact_score = (avg_high_10d - avg_low_10d) / sum_volume_10d
    
    # Generate Factor 1
    factor_1 = intraday_return * volume_impact_score
    
    # Calculate Cumulative Volume-Weighted Momentum
    cvwm = (intraday_return * df['volume']).rolling(window=25).sum()
    
    # Final Factor Calculation
    if vr > 1.4:
        final_factor = (price_momentum + intraday_momentum) * amt_adjusted
        aggregate_momentum = cvwm
    else:
        final_factor = 0.7 * (price_momentum + intraday_momentum + amt_adjusted)
        aggregate_momentum = 0.7 * cvwm
    
    # Final Factor
    final_factor = factor_1 * aggregate_momentum
    
    return final_factor
