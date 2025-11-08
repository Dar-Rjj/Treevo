import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Price Dynamics
    df['M2'] = df['close'] / df['close'].shift(2) - 1
    df['M5'] = df['close'] / df['close'].shift(5) - 1
    df['M13'] = df['close'] / df['close'].shift(13) - 1
    df['M34'] = df['close'] / df['close'].shift(34) - 1
    
    # Volatility Environment Assessment
    df['NDR'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['Vol13'] = df['NDR'].rolling(window=13, min_periods=13).std()
    df['Vol13_median_63d'] = df['Vol13'].rolling(window=63, min_periods=63).median()
    df['VolRegime'] = df['Vol13'] / df['Vol13_median_63d']
    
    # Volume Behavior Analysis
    df['VolMom'] = df['volume'] / df['volume'].shift(5) - 1
    df['VolPers'] = df['volume'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(1, 6), x.values)[0, 1] if len(x) == 5 else np.nan, 
        raw=False
    )
    df['VolRegime_vol'] = 1 / (1 + abs(df['VolMom'])) * (1 + df['VolPers'])
    
    # Momentum Integration & Adjustment
    df['VM2'] = df['M2'] / (df['Vol13'] + 0.0001)
    df['VM5'] = df['M5'] / (df['Vol13'] + 0.0001)
    df['VM13'] = df['M13'] / (df['Vol13'] + 0.0001)
    df['VM34'] = df['M34'] / (df['Vol13'] + 0.0001)
    
    df['MomentumConsistency'] = (np.sign(df['VM2']) + np.sign(df['VM5']) + 
                                np.sign(df['VM13']) + np.sign(df['VM34'])) / 4
    df['MomentumStrength'] = (abs(df['VM2']) + abs(df['VM5']) + 
                             abs(df['VM13']) + abs(df['VM34'])) / 4
    
    df['MomentumScore'] = df['MomentumConsistency'] * df['MomentumStrength'] * df['VolRegime_vol']
    
    # Final Alpha Construction
    alpha = df['MomentumScore'] * df['VolRegime_vol']
    
    return alpha
