import pandas as pd
import numpy as np

def heuristics_v2(df):
    high, low, close, open, volume, amount = df['high'], df['low'], df['close'], df['open'], df['volume'], df['amount']
    
    gap = (open - close.shift(1)) / close.shift(1)
    intraday_range = (high - low) / open
    volume_trend = volume / volume.rolling(5).mean()
    
    gap_rank = gap.rolling(10).apply(lambda x: pd.Series(x).rank().iloc[-1])
    range_rank = intraday_range.rolling(10).apply(lambda x: pd.Series(x).rank().iloc[-1])
    volume_rank = volume_trend.rolling(10).apply(lambda x: pd.Series(x).rank().iloc[-1])
    
    heuristics_matrix = -gap_rank * range_rank * volume_rank
    
    return heuristics_matrix
