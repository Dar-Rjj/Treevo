import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df.shift(1)['close']) / df.shift(1)['close']
    
    # Combine Intraday Range and Close-to-Open Return with weights
    recent_weights = 0.7 * intraday_range + 0.3 * close_to_open_return
    older_weights = 0.5 * intraday_range + 0.5 * close_to_open_return
    
    # Define a period to distinguish recent and older data, e.g., last 30 days
    recent_period = 30
    combined_factor = df['close'].copy()
    combined_factor.iloc[-recent_period:] = recent_weights.iloc[-recent_period:]
    combined_factor.iloc[:-recent_period] = older_weights.iloc[:-recent_period]
    
    # Calculate Volume-Weighted Average Price (VWAP)
    vwap = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Integrate VWAP with Intraday Volatility Adjusted Return
    final_factor = combined_factor * vwap
    
    return final_factor
