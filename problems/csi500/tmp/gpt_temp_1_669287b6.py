import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Adjust High-Low Range by Volume and Close Price
    df['avg_volume_30'] = df['volume'].rolling(window=30).mean()
    df['avg_close_30'] = df['close'].rolling(window=30).mean()
    df['adjusted_high_low_range'] = (df['high_low_range'] * (df['volume'] / df['avg_volume_30']) * 
                                     (df['close'] / df['avg_close_30']))
    
    # Compute Exponentially Weighted Moving Average of Recent Adjusted Ranges
    smoothing_factor = 0.2
    weights = [(1 - smoothing_factor) ** n for n in range(10)]
    adjusted_ranges = df['adjusted_high_low_range'].rolling(window=10, min_periods=1).apply(
        lambda x: np.dot(x, weights[:len(x)]), raw=True)
    df['high_low_range_ema'] = adjusted_ranges
    
    # Calculate Short-Term Return
    df['short_term_return'] = df['close'].pct_change(5)
    
    # Calculate Long-Term Return
    df['long_term_return'] = df['close'].pct_change(20)
    
    # Combine Momentum Indicators
    df['momentum_indicator'] = (df['long_term_return'] - df['short_term_return']) / df['volume']
    
    # Intraday Return
    df['intraday_return'] = (df['high'] - df['open']) / df['open']
    
    # Close-to-Open Reversal
    df['close_to_open_reversal'] = (df['close'] - df['open']) / df['open']
    
    # Volume Adjusted Intraday Volatility
    df['vol_adj_intraday_vol'] = (df['high'] - df['low']) * np.sqrt(df['volume'])
    
    # Calculate High-Low Range Ratio
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Integrate Additional Price Patterns
    def bullish_engulfing(row):
        return (row['close'] > row['open']) and (row['close_prev'] < row['open_prev']) and (row['close'] > row['open_prev']) and (row['close_prev'] < row['open'])

    def bearish_engulfing(row):
        return (row['close'] < row['open']) and (row['close_prev'] > row['open_prev']) and (row['close'] < row['open_prev']) and (row['close_prev'] > row['open'])

    def doji_pattern(row):
        return abs(row['close'] - row['open']) <= 0.005 * (row['high'] - row['low'])

    def hammer_pattern(row):
        return (row['close'] - row['low']) / (row['high'] - row['low']) > 0.6 and (row['high'] - row['close']) / (row['high'] - row['low']) < 0.4
    
    df['open_prev'] = df['open'].shift(1)
    df['close_prev'] = df['close'].shift(1)
    
    df['bullish_engulfing'] = df.apply(bullish_engulfing, axis=1)
    df['bearish_engulfing'] = df.apply(bearish_engulfing, axis=1)
    df['doji_pattern'] = df.apply(doji_pattern, axis=1)
    df['hammer_pattern'] = df.apply(hammer_pattern, axis=1)
    
    # Combine Pattern Scores
    df['pattern_score'] = (df['bullish_engulfing'] - df['bearish_engulfing'] + df['doji_pattern'] + df['hammer_pattern']) * 0.25
    
    # Final Indicator Combination
    df['volume_weighted_high_low_range'] = df['high_low_range'] * df['high_low_ratio'] * df['volume']
    df['open_close_return_adjustment'] = (df['close'] - df['open']) / df['open']
    df['final_alpha_factor'] = (df['high_low_range_ema'] * df['intraday_return'] * df['close_to_open_reversal'] * 
                                df['vol_adj_intraday_vol'] * df['pattern_score'] * df['open_close_return_adjustment'] * 
                                df['volume_weighted_high_low_range'])
    
    return df['final_alpha_factor']
