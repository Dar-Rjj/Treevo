import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Price reversal component (short-term mean reversion)
    price_reversal = -1 * (close - close.rolling(5).mean()) / close.rolling(5).std()
    
    # Volume confirmation (abnormal volume on price moves)
    volume_rank = volume.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    price_change = close.pct_change(3)
    volume_confirmation = volume_rank * np.sign(price_change)
    
    # Volatility-adjusted momentum (medium-term momentum normalized by volatility)
    vol_adj_momentum = (close / close.rolling(15).mean() - 1) / close.rolling(20).std()
    
    # Combined alpha factor
    heuristics_matrix = price_reversal + volume_confirmation + vol_adj_momentum
    
    return heuristics_matrix
