import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Price reversal component (short-term mean reversion)
    price_reversal = -close.pct_change(5)
    
    # Volume confirmation (abnormal volume during reversal)
    volume_rank = volume.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    volume_confirmation = price_reversal * volume_rank
    
    # Volatility regime adjustment (low vol periods get higher weights)
    volatility = close.pct_change().rolling(20).std()
    vol_regime = 1 / (1 + volatility)
    
    # Price momentum quality (smoothness of trend)
    mom_quality = close.rolling(5).apply(lambda x: np.corrcoef(range(5), x)[0,1] if len(x) == 5 else np.nan)
    
    # Liquidity component (amount-based)
    vwap = amount / volume
    liquidity_signal = (close - vwap) / vwap
    
    # Combine components
    heuristics_matrix = (price_reversal * 0.4 + 
                        volume_confirmation * 0.3 + 
                        vol_regime * 0.15 + 
                        mom_quality * 0.1 + 
                        liquidity_signal * 0.05)
    
    return heuristics_matrix
