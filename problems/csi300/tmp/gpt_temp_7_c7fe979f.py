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
    overnight_gap = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)
    price_volatility = (high - low) / close.replace(0, np.nan)
    
    factor = (intraday_momentum * volume_efficiency) - (overnight_gap * price_volatility)
    heuristics_matrix = factor
    
    return heuristics_matrix
