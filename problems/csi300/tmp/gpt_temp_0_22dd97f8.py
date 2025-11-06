import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Correlation-based trend persistence
    returns_5 = close.pct_change(5)
    returns_10 = close.pct_change(10)
    trend_persistence = returns_5.rolling(15).corr(returns_10)
    
    # Entropy-weighted price efficiency
    price_range = (high - low) / ((high + low) / 2)
    range_entropy = -price_range.rolling(10).apply(lambda x: (x/x.sum() * np.log(x/x.sum())).sum(), raw=False)
    efficiency_entropy = price_range * range_entropy
    
    # Residual-driven liquidity momentum
    vwap = amount / volume
    price_residual = close - vwap.rolling(5).mean()
    liquidity_momentum = price_residual.diff(3) * volume.rolling(8).mean()
    
    # Combine components
    heuristics_matrix = trend_persistence.rank(pct=True) + efficiency_entropy.rank(pct=True) + liquidity_momentum.rank(pct=True)
    
    return heuristics_matrix
