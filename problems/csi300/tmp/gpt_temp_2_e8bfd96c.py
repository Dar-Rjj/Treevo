import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Volatility and High-Low Range
    intraday_volatility = (df['high'] - df['low'])
    high_low_range = (df['high'] - df['low']).rolling(window=5).mean()  # Lookback period of 5 days
    
    # Weight by Volume-to-Price Ratio
    average_price = (df['high'] + df['low']) / 2
    volume_to_price_ratio = df['volume'] / average_price
    weighted_intraday_volatility = intraday_volatility * volume_to_price_ratio
    
    # Enhance with Close-to-Open Change
    close_open_change = (df['close'] - df['open']).abs()
    enhanced_close_open_change = weighted_intraday_volatility - close_open_change
    
    # Calculate Momentum
    high_low_momentum = high_low_range.rolling(window=10).mean()  # Lookback period of 10 days
    close_open_spread_momentum = (df['close'] - df['open']).rolling(window=10).mean()  # Lookback period of 10 days
    
    # Compute Final Factor
    combined_factor = enhanced_close_open_change
    adjusted_by_momentum = combined_factor.copy()
    
    for i in range(10, len(df)):  # Start from the 10th day to avoid NaNs from the rolling window
        if close_open_spread_momentum[i] > 0:
            adjusted_by_momentum[i] /= close_open_spread_momentum[i]
        else:
            adjusted_by_momentum[i] *= abs(close_open_spread_momentum[i])
    
    return pd.Series(adjusted_by_momentum, index=df.index)
