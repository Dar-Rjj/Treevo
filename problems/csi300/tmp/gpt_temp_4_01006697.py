import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Compute Previous Day's Open-to-Close Return
    previous_day_open = df['open'].shift(1)
    today_close = df['close']
    prev_open_to_today_close_return = today_close - previous_day_open
    
    # Calculate Volume Weighted Average Price (VWAP)
    daily_vwap = (df[['high', 'low', 'close', 'open']].mean(axis=1) * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Combine Intraday Momentum and VWAP
    combined_value = daily_vwap - intraday_high_low_spread
    weighted_combined_value = combined_value * df['volume']
    
    # Smooth the Factor
    smoothed_factor = weighted_combined_value.ewm(span=7).mean()
    
    # Apply Adaptive EMA
    returns = df['close'].pct_change()
    recent_volatility = returns.rolling(window=7).std()
    smoothing_factor = 1 / (1 + recent_volatility)
    adaptive_ema = smoothed_factor.ewm(alpha=smoothing_factor, adjust=False).mean()
    
    return adaptive_ema
