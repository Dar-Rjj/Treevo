import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    
    overnight_gap = (open_ - close.shift(1)) / close.shift(1)
    morning_range = (high - open_) / open_
    volume_intensity = volume / volume.rolling(window=5).mean()
    
    gap_reversal = -overnight_gap * morning_range * volume_intensity
    heuristics_matrix = gap_reversal
    
    return heuristics_matrix
