import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Adjust Intraday Return by Intraday Range
    df['adjusted_intraday_return'] = df['intraday_return'] / df['intraday_range']
    
    # Incorporate Volume Trend
    df['volume_change'] = df['volume'] / df['volume'].shift(1)
    df['volume_adjusted_intraday_return'] = df['adjusted_intraday_return'] * df['volume_change']
    
    # Calculate Volume-Weighted Average Price (VWAP)
    df['vwap'] = ((df['high'] + df['low']) / 2 * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate Daily Return using VWAP
    df['daily_return'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Smooth the Daily Return
    span = 5
    df['smoothed_daily_return'] = df['daily_return'].ewm(span=span, adjust=False).mean()
    
    # Combine Factors
    df['combined_factor_1'] = df['volume_adjusted_intraday_return'] + df['smoothed_daily_return']
    
    # Calculate High-to-Low Range
    df['high_to_low_range'] = df[['high', 'low']].max(axis=1) - df[['high', 'low']].min(axis=1)
    
    # Normalize by Open Price
    df['normalized_high_to_low_range'] = df[['high - open', 'open - low']].max(axis=1) / df['open']
    df['high - open'] = df['high'] - df['open']
    df['open - low'] = df['open'] - df['low']
    
    # Incorporate Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['open']
    df['normalized_high_to_low_range_volatility'] = df['normalized_high_to_low_range'] / df['intraday_volatility']
    
    # Combine Final Factors
    df['final_alpha_factor'] = df['combined_factor_1'] + df['normalized_high_to_low_range_volatility']
    
    return df['final_alpha_factor']
