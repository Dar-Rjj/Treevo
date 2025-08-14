import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_range = df['high'] - df['low']
    
    # Adjust Intraday High-Low Spread by Volume
    volume_ema_span = 10
    volume_ema = df['volume'].ewm(span=volume_ema_span, adjust=False).mean()
    volume_adjusted_intraday_range = intraday_range * (df['volume'] / volume_ema)
    
    # Further Adjustment by Open Price Volatility
    open_price_volatility = df['open'].pct_change().rolling(window=20).std()
    adjusted_intraday_range = volume_adjusted_intraday_range / open_price_volatility
    
    # Calculate Short-Term Volume Adjusted Return
    short_term_return = (df['close'].pct_change() * df['volume']).rolling(window=5).mean()
    
    # Calculate Long-Term Volume Adjusted Return
    long_term_return = (df['close'].pct_change() * df['volume']).rolling(window=20).mean()
    
    # Aggregate Intraday Volatility
    intraday_volatility = (adjusted_intraday_range + (df['close'] - df['open'])) * df['volume']
    intraday_volatility_ma = intraday_volatility.rolling(window=20).mean()
    
    # Calculate Relative Strength Indicator (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Price Reversal Indicator
    local_highs = df['close'].rolling(window=10).max()
    local_lows = df['close'].rolling(window=10).min()
    time_since_high = (df.index.to_series() - df[df['close'] == local_highs].index.to_series()).dt.days
    time_since_low = (df.index.to_series() - df[df['close'] == local_lows].index.to_series()).dt.days
    reversal_indicator = (time_since_high - time_since_low) / (time_since_high + time_since_low + 1)
    
    # Combine Factors for Final Alpha
    return_difference = short_term_return - long_term_return
    final_alpha = (return_difference 
                   + (intraday_volatility_ma * 0.3)
                   + (rsi - 50) * 0.2
                   + (reversal_indicator * 0.5))
    
    return final_alpha
