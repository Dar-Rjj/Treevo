import pandas as pd

def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    
    price_momentum = close.pct_change(periods=10)
    volume_trend = volume.rolling(window=10).apply(lambda x: pd.Series(x).pct_change().mean())
    
    directional_consistency = (close.diff() > 0).rolling(window=10).mean() - 0.5
    
    divergence_factor = price_momentum - volume_trend
    heuristics_matrix = divergence_factor * directional_consistency
    
    return heuristics_matrix
