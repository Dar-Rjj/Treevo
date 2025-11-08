import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_rank = close.rolling(window=21).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    volume_cluster = volume.rolling(window=8).std() / volume.rolling(window=21).std()
    volatility_regime = (high - low).rolling(window=5).std() / (high - low).rolling(window=21).std()
    
    mean_reversion = (0.5 - price_rank) * volume_cluster
    regime_adjusted = mean_reversion * volatility_regime
    
    heuristics_matrix = pd.Series(regime_adjusted, index=df.index, name='heuristics_v2')
    return heuristics_matrix
