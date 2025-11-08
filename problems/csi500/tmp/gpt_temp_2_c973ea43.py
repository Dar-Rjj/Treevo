import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Calculation
    df['M1'] = df['close'] / df['close'].shift(1) - 1
    df['M2'] = df['close'] / df['close'].shift(2) - 1
    df['M3'] = df['close'] / df['close'].shift(3) - 1
    
    # Volatility Context
    df['Vol'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Volume Analysis
    def calculate_volume_percentile(series):
        if len(series) < 21:
            return np.nan
        current_vol = series.iloc[-1]
        lookback_vol = series.iloc[:-1]
        return (lookback_vol <= current_vol).sum() / len(lookback_vol)
    
    df['VolPct'] = df['volume'].rolling(window=21).apply(calculate_volume_percentile, raw=False)
    df['VolStability'] = 1 / (1 + abs(df['VolPct'] - 0.5))
    
    # Volatility-Weighted Momentum
    df['VWM1'] = df['M1'] / (df['Vol'] + 0.0001)
    df['VWM2'] = df['M2'] / (df['Vol'] + 0.0001)
    df['VWM3'] = df['M3'] / (df['Vol'] + 0.0001)
    
    # Geometric Momentum Convergence
    df['SignAgreement'] = np.sign(df['VWM1']) * np.sign(df['VWM2']) * np.sign(df['VWM3'])
    df['GeoMomentum'] = (abs(df['VWM1'] * df['VWM2'] * df['VWM3'])) ** (1/3)
    df['MomentumConvergence'] = df['SignAgreement'] * df['GeoMomentum']
    
    # Final Alpha Signal
    alpha = df['MomentumConvergence'] * df['VolStability']
    
    return alpha
