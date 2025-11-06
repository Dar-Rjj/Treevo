import pandas as pd

def heuristics_v2(df):
    high, low, close, volume, amount = df['high'], df['low'], df['close'], df['volume'], df['amount']
    
    intraday_momentum = (close - (high + low) / 2) / (high - low + 1e-12)
    volume_efficiency = (high - low) / (amount / volume + 1e-12)
    price_volatility = (high - low) / close
    turnover_asymmetry = (volume - volume.rolling(5).mean()) / (volume.rolling(20).mean() + 1e-12)
    
    combined_factor = (intraday_momentum * volume_efficiency) - (price_volatility * turnover_asymmetry)
    normalized_factor = (combined_factor - combined_factor.rolling(10).mean()) / (combined_factor.rolling(20).std() + 1e-12)
    heuristics_matrix = normalized_factor.rolling(3).mean()
    
    return heuristics_matrix
