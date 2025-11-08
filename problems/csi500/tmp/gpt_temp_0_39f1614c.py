import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # Idiosyncratic momentum with volatility regime filtering
    market_returns = close.pct_change()
    stock_returns = close.pct_change()
    rolling_beta = stock_returns.rolling(window=20).cov(market_returns) / market_returns.rolling(window=20).var()
    residual_returns = stock_returns - rolling_beta * market_returns
    vol_regime = stock_returns.rolling(window=30).std()
    regime_adjusted_momentum = residual_returns.rolling(window=15).mean() / vol_regime
    
    # Volume-weighted reversal around earnings events (proxy via high volume days)
    volume_rank = volume.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    price_reversal = -close.pct_change(3)
    earnings_reversal_signal = price_reversal * volume_rank
    
    # Price trend efficiency (smoothness of price movement)
    price_range = high.rolling(window=10).max() - low.rolling(window=10).min()
    price_movement = close.diff(10).abs()
    trend_efficiency = price_movement / price_range
    
    # Combined alpha factor
    heuristics_matrix = regime_adjusted_momentum + earnings_reversal_signal + trend_efficiency
    
    return heuristics_matrix
