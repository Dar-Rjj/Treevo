import pandas as pd
import numpy as np

def heuristics_v2(df):
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Relative strength mean reversion component
    rs_ratio = close / close.rolling(window=10).mean()
    price_rank = (close.rank() - close.rolling(window=20).apply(lambda x: x.rank().iloc[-1])) / 20
    
    # Volatility-scaled liquidity absorption
    dollar_volume = close * volume
    vol_adjusted_flow = dollar_volume / dollar_volume.rolling(window=15).std()
    absorption_ratio = (high - close) / (high - low + 1e-8) * vol_adjusted_flow
    
    # Regime-sensitive momentum filter
    market_regime = close.rolling(window=10).std() / close.rolling(window=30).std()
    regime_momentum = close.pct_change(5) * (2 - market_regime)
    
    # Composite factor construction
    heuristics_matrix = (rs_ratio - 1) * price_rank + absorption_ratio - regime_momentum
    
    return heuristics_matrix
