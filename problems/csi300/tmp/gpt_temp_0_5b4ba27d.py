import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Fractal Regime Classification
    df['volatility_fractal'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    df['volume_fractal'] = df['volume'] / df['volume'].shift(1)
    df['trend_fractal'] = np.sign(df['close'] - df['close'].shift(5)) * np.sign(df['close'].shift(5) - df['close'].shift(20))
    
    # Asymmetric Microstructure
    df['upside_rejection'] = df['high'] - np.maximum(df['open'], df['close'])
    df['downside_rejection'] = np.minimum(df['open'], df['close']) - df['low']
    df['net_asymmetry'] = df['upside_rejection'] - df['downside_rejection']
    
    # Efficiency Dynamics
    df['intraday_efficiency'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['intraday_efficiency'] = df['intraday_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Integration
    df['volume_range_coherence'] = (df['volume'] * (df['high'] - df['low'])) / (df['volume'].shift(1) * (df['high'].shift(1) - df['low'].shift(1)))
    
    # Breakout Patterns
    df['clean_momentum'] = df['close'] / df['close'].shift(1) - 1
    
    # Cross-Fractal Validation
    df['efficiency_volume_divergence'] = np.sign(df['intraday_efficiency'] - df['intraday_efficiency'].shift(1)) * np.sign(df['volume'] / df['volume'].shift(1) - 1)
    
    # Momentum Consistency (rolling calculation for past 3 periods)
    momentum_sign = np.sign(df['clean_momentum'])
    momentum_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(2, len(df)):
        if i >= 2:
            window = momentum_sign.iloc[i-2:i+1]
            consistent_count = (window == window.shift(1)).sum() - 1  # Subtract 1 because first comparison is NaN
            momentum_consistency.iloc[i] = consistent_count / 3
    df['momentum_consistency'] = momentum_consistency
    
    # Alpha Synthesis
    df['microstructure_velocity'] = (df['net_asymmetry'] * 
                                   (df['intraday_efficiency'] / df['intraday_efficiency'].shift(1) - 1) * 
                                   df['efficiency_volume_divergence'])
    
    df['composite_alpha'] = (df['microstructure_velocity'] * 
                           df['volume_range_coherence'] * 
                           df['momentum_consistency'])
    
    return df['composite_alpha']
