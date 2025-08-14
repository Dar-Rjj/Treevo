import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator

def heuristics_v2(df, n_days=10):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Open-Close Range
    open_close_range = df['close'] - df['open']
    
    # Calculate Relative Strength Indicator (RSI)
    rsi = RSIIndicator(df['close'], window=14).rsi()
    
    # Aggregate Intraday Volatility
    average_volume = df['volume'].rolling(window=n_days).mean()
    volume_weighted_intraday_volatility = (intraday_high_low_spread * df['volume']) / average_volume
    
    # Apply Exponential Moving Average to Intraday Volatility
    ema_intraday_volatility = volume_weighted_intraday_volatility.ewm(span=n_days).mean()
    
    # Adjust RSI with Volume-Weighted Average
    adjusted_rsi = rsi * (df['volume'] / average_volume)
    
    # Summarize Combined Weights
    combined_weights = ema_intraday_volatility + adjusted_rsi + open_close_range
    
    # Integrate Trade Amount Impact
    trade_amount_difference = df['amount'] - df['amount'].shift(1)
    normalized_trade_amount_difference = trade_amount_difference / df['volume']
    
    # Incorporate into Final Alpha Factor
    final_alpha_factor = combined_weights + normalized_trade_amount_difference
    
    return final_alpha_factor
