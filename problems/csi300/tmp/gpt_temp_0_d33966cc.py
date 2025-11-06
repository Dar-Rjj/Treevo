import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    amount = df['amount']
    
    intraday_momentum = (close - open_) / (high - low).replace(0, np.nan)
    volume_efficiency = amount / volume.replace(0, np.nan)
    price_volatility = (high - low) / close.shift(1).replace(0, np.nan)
    overnight_gap_persistence = (open_ - close.shift(1)).abs() / (high - low).replace(0, np.nan)
    
    numerator = intraday_momentum * volume_efficiency
    denominator = price_volatility * overnight_gap_persistence
    
    heuristics_matrix = numerator - denominator
    heuristics_matrix = heuristics_matrix.replace([np.inf, -np.inf], np.nan)
    
    return heuristics_matrix
