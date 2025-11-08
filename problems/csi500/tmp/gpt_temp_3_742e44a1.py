import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Horizon-Matched Volatility-Adjusted Momentum with Signal Agreement
    """
    # Multi-Horizon Momentum Calculation
    df['M2'] = df['close'] / df['close'].shift(2) - 1
    df['M5'] = df['close'] / df['close'].shift(5) - 1
    df['M13'] = df['close'] / df['close'].shift(13) - 1
    df['M34'] = df['close'] / df['close'].shift(34) - 1
    
    # Daily Range Calculation
    df['Range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Rolling Volatility by Horizon
    df['Vol2'] = df['Range'].rolling(window=2).std()
    df['Vol5'] = df['Range'].rolling(window=5).std()
    df['Vol13'] = df['Range'].rolling(window=13).std()
    df['Vol34'] = df['Range'].rolling(window=34).std()
    
    # Volatility-Adjusted Momentum
    df['M2_vol'] = df['M2'] / (df['Vol2'] + 1e-8)
    df['M5_vol'] = df['M5'] / (df['Vol5'] + 1e-8)
    df['M13_vol'] = df['M13'] / (df['Vol13'] + 1e-8)
    df['M34_vol'] = df['M34'] / (df['Vol34'] + 1e-8)
    
    # Signal Agreement Analysis
    momentum_signals = pd.DataFrame({
        'M2_pos': df['M2_vol'] > 0,
        'M5_pos': df['M5_vol'] > 0,
        'M13_pos': df['M13_vol'] > 0,
        'M34_pos': df['M34_vol'] > 0
    })
    
    agreement_count = momentum_signals.sum(axis=1)
    
    # Agreement-Based Weighting
    agreement_weights = pd.Series(index=df.index, dtype=float)
    agreement_weights[agreement_count == 4] = 1.0
    agreement_weights[agreement_count == 3] = 0.8
    agreement_weights[agreement_count == 2] = 0.5
    agreement_weights[agreement_count <= 1] = 0.2
    
    # Volume Confirmation
    df['VolumePercentile'] = df['volume'].rolling(window=20).apply(
        lambda x: (x.rank(pct=True).iloc[-1] * 100), raw=False
    )
    df['VolConf'] = (df['VolumePercentile'] / 100) ** 0.25
    
    # Weighted Momentum Integration
    df['WM2'] = df['M2_vol'] * agreement_weights
    df['WM5'] = df['M5_vol'] * agreement_weights
    df['WM13'] = df['M13_vol'] * agreement_weights
    df['WM34'] = df['M34_vol'] * agreement_weights
    
    # Horizon-Weighted Combination
    df['MomentumBlend'] = (
        0.3 * df['WM2'] + 
        0.3 * df['WM5'] + 
        0.25 * df['WM13'] + 
        0.15 * df['WM34']
    )
    
    # Final Alpha Construction
    alpha = df['MomentumBlend'] * df['VolConf']
    
    return alpha.dropna()
