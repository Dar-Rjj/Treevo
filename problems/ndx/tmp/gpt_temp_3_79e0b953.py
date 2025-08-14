import pandas as pd
import pandas as pd

def heuristics_v2(df, lookback=20):
    # Calculate the rate of change (ROC) of the close price over a lookback period
    df['roc_close'] = df['close'].pct_change(periods=lookback)
    
    # Determine the percentage difference between the highest high and lowest low over a lookback period
    df['high_roll_max'] = df['high'].rolling(window=lookback).max()
    df['low_roll_min'] = df['low'].rolling(window=lookback).min()
    df['pct_diff_high_low'] = (df['high_roll_max'] - df['low_roll_min']) / df['low_roll_min']
    
    # Create a cumulative sum of volume adjusted by close price movement
    df['volume_adjusted'] = df['volume'] * df['close'].diff() / df['close'].shift(1)
    df['cumulative_volume'] = df['volume_adjusted'].rolling(window=lookback).sum()
    
    # Calculate the ratio of volume on days where close is above the previous day's close to total volume over a lookback period
    df['volume_up'] = df.apply(lambda row: row['volume'] if row['close'] > row['close'].shift(1) else 0, axis=1)
    df['volume_ratio'] = df['volume_up'].rolling(window=lookback).sum() / df['volume'].rolling(window=lookback).sum()
    
    # Generate a volume-weighted moving average (VWMA) of the closing price
    df['vwap'] = (df['close'] * df['volume']).rolling(window=lookback).sum() / df['volume'].rolling(window=lookback).sum()
    
    # Detect bullish and bearish engulfing patterns with significant volume
    df['bullish_engulfing'] = ((df['open'] > df['close'].shift(1)) & 
                               (df['close'] >= df['open'].shift(1)) & 
                               (df['close'] > df['open']) & 
                               (df['volume'] > df['volume'].shift(1)))
    df['bearish_engulfing'] = ((df['open'] < df['close'].shift(1)) & 
                               (df['close'] <= df['open'].shift(1)) & 
                               (df['close'] < df['open']) & 
                               (df['volume'] > df['volume'].shift(1)))
    
    # Measure the strength of a trend by comparing the distance between the current close and the VWMA
    df['trend_strength'] = df['close'] - df['vwap']
    
    # Combine all factors into a single alpha factor
    alpha_factor = (df['roc_close'] + 
                    df['pct_diff_high_low'] + 
                    df['cumulative_volume'] + 
                    df['volume_ratio'] + 
                    df['trend_strength'])
    
    # Normalize the alpha factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=lookback).mean()) / alpha_factor.rolling(window=lookback).std()
    
    return alpha_factor
