import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['High'] - df['Low']

    # Compute Previous Day's Open-to-Close Return
    previous_open = df['Open'].shift(1)
    open_to_close_return = df['Close'] - previous_open

    # Calculate Volume Weighted Average Price (VWAP)
    vwap = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum() / df['Volume'].cumsum()

    # Combine Intraday Momentum and VWAP
    combined_value = vwap - intraday_high_low_spread

    # Weight by Intraday Volume
    volume_weighted_combined_value = combined_value * df['Volume']

    # Smooth the Factor with Exponential Moving Average (EMA)
    smoothed_factor = volume_weighted_combined_value.ewm(span=7, adjust=False).mean()

    # Consider Impact of Volume on Intraday Momentum
    volume_adjusted_intraday_momentum = open_to_close_return * (df['Volume'] / df['Volume'].rolling(window=7).mean())
    
    # Apply Short-Term EMA to Volume-Adjusted Intraday Momentum
    smoothed_volume_adjusted_intraday_momentum = volume_adjusted_intraday_momentum.ewm(span=7, adjust=False).mean()

    # Final alpha factor
    final_factor = (smoothed_factor + smoothed_volume_adjusted_intraday_momentum) / 2

    return final_factor
