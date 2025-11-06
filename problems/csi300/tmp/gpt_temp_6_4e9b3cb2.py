import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Volatility Regime Core
    df['vol_range'] = df['high'] - df['low']
    df['vol_momentum_asymmetry'] = (df['vol_range'] / df['vol_range'].shift(1)) * np.sign(df['close'] - df['close'].shift(1)) * (df['close'] - df['close'].shift(1))
    
    df['vol_divergence'] = (df['vol_range'] / df['vol_range'].shift(5)) - (df['vol_range'].shift(2) / df['vol_range'].shift(7)) * np.sign(df['close'] - df['close'].shift(2))
    
    vol_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(df)):
        if i >= 3:
            window = df.iloc[i-3:i+1]
            count = ((window['vol_range'] > window['vol_range'].shift(1)).sum() - 1) / 4
            vol_persistence.iloc[i] = count
    
    # Volume-Flow Integration
    df['volume_coherence'] = (df['volume'] - df['volume'].shift(2)) * df['vol_range'] / (np.abs(df['volume'] - df['volume'].shift(2)) + 1e-8)
    
    df['flow_absorption'] = (df['close'] - df['open']) * df['volume'] - (df['high'] - df['close']) * df['volume'].shift(1)
    
    df['vol_weighted_flow'] = np.abs(df['vol_range']) * df['volume'] / (np.abs(df['close'] - df['close'].shift(2)) + 1e-8)
    
    # Range Dynamics
    df['range_absorption'] = (df['vol_range'] / df['vol_range'].shift(1)) * (df['close'] - df['open']) / (np.abs(df['close'] - df['open']) + 1e-8)
    
    df['range_flow_regime'] = df['vol_range'] * df['volume'] / (np.abs(df['vol_range']) + 1e-8) * df['range_absorption']
    
    df['range_efficiency'] = (np.abs(df['close'] - df['open']) / (df['vol_range'] + 1e-8)) * (np.abs(df['vol_range']) / (np.abs(df['close'] - df['close'].shift(1)) + 1e-8))
    
    # Volatility Framework
    df['intraday_vol_ratio'] = df['vol_range'] / (np.abs(df['close'] - df['close'].shift(1)) + 1e-8)
    
    df['multi_day_vol'] = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()) / (np.abs(df['close'] - df['close'].shift(4)) + 1e-8)
    
    df['vol_asymmetry_signal'] = df['intraday_vol_ratio'] * (1 - df['multi_day_vol']) * np.sign(df['close'] - df['close'].shift(2))
    
    # Momentum Integration
    df['vol_momentum'] = (df['close'] - df['close'].shift(2)) * df['vol_range'] / (np.abs(df['close'] - df['close'].shift(2)) + 1e-8)
    
    df['multi_day_momentum_fractal'] = (df['close'] - df['close'].shift(4)) * df['vol_range'] / (np.abs(df['close'] - df['close'].shift(4)) + 1e-8)
    
    momentum_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        if i >= 4:
            window = df.iloc[i-4:i+1]
            count = 0
            for j in range(1, len(window)-2):
                if window['vol_momentum'].iloc[j] * window['vol_momentum'].iloc[j-1] > 0:
                    count += 1
            momentum_persistence.iloc[i] = count / 3
    
    # Alpha Synthesis
    df['core_volatility'] = df['vol_momentum_asymmetry'] * vol_persistence
    df['flow_efficiency'] = df['volume_coherence'] * df['vol_weighted_flow']
    df['range_integration'] = df['range_flow_regime'] * df['range_efficiency']
    df['momentum_enhancement'] = df['vol_momentum'] * momentum_persistence
    
    # Composite Alpha
    alpha = df['core_volatility'] * df['flow_efficiency'] * df['range_integration'] * df['momentum_enhancement']
    
    return alpha.fillna(0)
