import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Volatility-adjusted momentum
    ret_5 = close.pct_change(5)
    vol_20 = close.pct_change().rolling(20).std()
    vol_adj_momentum = ret_5 / (vol_20 + 1e-8)
    
    # Volume-confirmed reversal
    price_range = (high - low) / close
    volume_rank = volume.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    reversal_signal = -ret_5 * volume_rank * price_range
    
    # Liquidity-based scaling
    vwap = amount / (volume + 1e-8)
    liquidity_premium = (close - vwap) / close
    turnover = volume / volume.rolling(20).mean()
    
    # Combine components
    heuristics_matrix = vol_adj_momentum + reversal_signal + liquidity_premium * turnover
    
    return heuristics_matrix
