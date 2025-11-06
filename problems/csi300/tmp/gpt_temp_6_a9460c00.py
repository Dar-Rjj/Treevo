import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    vwap = amount / (volume + 1e-5)
    short_term_reversal = -close.pct_change(3) * np.log(volume / volume.rolling(10).mean() + 1e-5)
    
    order_flow_pressure = (close - vwap) / (high - low + 1e-5)
    flow_persistence = order_flow_pressure.rolling(8).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    vol_regime = close.pct_change().rolling(15).std() / close.pct_change().rolling(50).std()
    regime_switch = np.where(vol_regime > vol_regime.rolling(10).mean(), 1, -1)
    
    asymmetric_response = np.where(close > close.rolling(20).mean(), 
                                  flow_persistence * short_term_reversal,
                                  short_term_reversal / (np.abs(flow_persistence) + 1e-5))
    
    heuristics_matrix = regime_switch * asymmetric_response * np.tanh(vol_regime)
    return heuristics_matrix
