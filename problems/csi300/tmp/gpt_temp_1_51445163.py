import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Price reversal component (short-term mean reversion)
    price_reversal = (close - close.rolling(5).mean()) / close.rolling(10).std()
    
    # Volume confirmation (abnormal volume during reversal)
    volume_rank = volume.rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    volume_confirmation = price_reversal * volume_rank
    
    # Volatility-normalized price range
    volatility = high.rolling(10).std()
    normalized_range = (high - low) / volatility
    
    # Amount-based liquidity factor
    vwap = amount / volume
    price_vwap_deviation = (close - vwap) / close.rolling(10).std()
    
    # Combined heuristic factor
    heuristics_matrix = (price_reversal * 0.3 + 
                        volume_confirmation * 0.4 + 
                        normalized_range * 0.2 + 
                        price_vwap_deviation * 0.1)
    
    return heuristics_matrix
