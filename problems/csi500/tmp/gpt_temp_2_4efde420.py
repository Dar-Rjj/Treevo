import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    volume_quantile = volume.rolling(20).apply(lambda x: pd.qcut(x, 2, labels=False, duplicates='drop').iloc[-1] if pd.notna(x.iloc[-1]) else np.nan, raw=False)
    high_volume_returns = close.pct_change().where(volume_quantile == 1)
    low_volume_returns = close.pct_change().where(volume_quantile == 0)
    
    efficiency_ratio = (high_volume_returns.rolling(5).std() / low_volume_returns.rolling(5).std()) * (high_volume_returns.rolling(5).mean() / low_volume_returns.rolling(5).mean())
    directional_consistency = (close.rolling(3).corr(volume.rolling(3).mean()) - close.rolling(10).corr(volume.rolling(10).mean()))
    volatility_compression = high.rolling(5).std() / low.rolling(5).std()
    
    heuristics_matrix = efficiency_ratio.rank() * directional_consistency.rank() - volatility_compression.rank()
    
    return heuristics_matrix
