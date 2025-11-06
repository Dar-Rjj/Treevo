import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    intraday_momentum = (close - (high + low) / 2) / (high - low + 1e-8)
    volume_efficiency = amount / (volume * close + 1e-8)
    price_volatility = (high - low) / (close + 1e-8)
    turnover_asymmetry = (volume - volume.shift(1)) / (volume.shift(1) + 1e-8)
    
    numerator = intraday_momentum * volume_efficiency
    denominator = price_volatility * turnover_asymmetry.abs()
    
    alpha_raw = (numerator - denominator) / (numerator.abs() + denominator.abs() + 1e-8)
    heuristics_matrix = alpha_raw - alpha_raw.rolling(10, min_periods=5).mean()
    
    return heuristics_matrix
