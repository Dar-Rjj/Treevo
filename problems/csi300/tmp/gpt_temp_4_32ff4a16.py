import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Scale Fractal Volatility Framework
    df['fractal_vol_3d'] = np.log(df['high'].rolling(window=3).apply(lambda x: (x - df.loc[x.index, 'low']).sum())) / np.log(3)
    df['fractal_vol_5d'] = np.log(df['high'].rolling(window=5).apply(lambda x: (x - df.loc[x.index, 'low']).sum())) / np.log(5)
    df['fractal_vol_10d'] = np.log(df['high'].rolling(window=10).apply(lambda x: (x - df.loc[x.index, 'low']).sum())) / np.log(10)
    
    df['fractal_vol_persistence'] = df['fractal_vol_5d'] / df['fractal_vol_10d']
    df['fractal_vol_acceleration'] = df['fractal_vol_3d'] / df['fractal_vol_5d']
    
    # Fractal Regime Classification
    high_fractal = (df['fractal_vol_3d'] > df['fractal_vol_5d']) & (df['fractal_vol_5d'] > df['fractal_vol_10d'])
    low_fractal = (df['fractal_vol_3d'] < df['fractal_vol_5d']) & (df['fractal_vol_5d'] < df['fractal_vol_10d'])
    transitional_fractal = ~high_fractal & ~low_fractal
    
    # Multi-Fractal Momentum with Volatility Sensitivity
    df['fractal_momentum_3d'] = (df['close'] / df['close'].shift(3)) - 1
    df['fractal_momentum_5d'] = (df['close'] / df['close'].shift(5)) - 1
    df['fractal_momentum_10d'] = (df['close'] / df['close'].shift(10)) - 1
    
    # Fractal Volatility-Weighted Momentum
    momentum_weights = pd.Series(index=df.index, dtype=float)
    momentum_weights[high_fractal] = 1 + df.loc[high_fractal, 'fractal_vol_persistence']
    momentum_weights[low_fractal] = 1 - df.loc[low_fractal, 'fractal_vol_persistence']
    momentum_weights[transitional_fractal] = df.loc[transitional_fractal, 'fractal_vol_acceleration']
    
    df['fractal_weighted_momentum'] = df['fractal_momentum_5d'] * momentum_weights
    
    # Fractal Volume-Efficiency Volatility Coherence
    df['fractal_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['fractal_efficiency_persistence'] = df['fractal_efficiency'].rolling(window=5).mean()
    
    df['volume_fractal_momentum'] = np.log(df['volume'] / df['volume'].shift(5))
    df['volume_fractal_concentration'] = np.log(df['volume'] / df['volume'].rolling(window=20).mean())
    
    # Amount-Volume Fractal Alignment Analysis
    df['amount_fractal_momentum'] = np.log(df['amount'] / df['amount'].shift(5))
    df['amount_fractal_concentration'] = np.log(df['amount'] / df['amount'].rolling(window=20).mean())
    
    # Multi-Dimensional Fractal Regime Confirmation
    df['fractal_component_agreement'] = (
        (df['fractal_vol_3d'].rolling(window=5).std() / df['fractal_vol_3d'].rolling(window=5).mean()) +
        (df['volume_fractal_momentum'].rolling(window=5).std() / abs(df['volume_fractal_momentum'].rolling(window=5).mean())) +
        (df['fractal_efficiency'].rolling(window=5).std() / df['fractal_efficiency'].rolling(window=5).mean())
    ) / 3
    
    # Adaptive Fractal Factor Synthesis Engine
    df['momentum_fractal_factor'] = df['fractal_weighted_momentum'] * df['fractal_efficiency_persistence']
    df['volume_fractal_factor'] = df['volume_fractal_concentration'] * df['volume_fractal_momentum']
    df['amount_fractal_factor'] = df['amount_fractal_concentration'] * df['amount_fractal_momentum']
    
    # Final Fractal Alpha Construction
    alpha = (
        df['momentum_fractal_factor'] * 
        df['volume_fractal_factor'] * 
        df['amount_fractal_factor'] * 
        (1 - df['fractal_component_agreement'])
    )
    
    # Apply regime-specific adjustments
    alpha[high_fractal] = alpha[high_fractal] * (1 + df.loc[high_fractal, 'fractal_vol_persistence'])
    alpha[low_fractal] = alpha[low_fractal] * (1 - df.loc[low_fractal, 'fractal_vol_persistence'])
    alpha[transitional_fractal] = alpha[transitional_fractal] * df.loc[transitional_fractal, 'fractal_vol_acceleration']
    
    return alpha
