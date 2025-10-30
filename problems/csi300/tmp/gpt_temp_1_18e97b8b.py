import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Efficiency Ratio
    # 5-day high-low efficiency
    df['eff_5d'] = (df['close'] - df['low'].shift(5)) / (df['high'].shift(5) - df['low'].shift(5))
    
    # 10-day high-low efficiency
    df['eff_10d'] = (df['close'] - df['low'].shift(10)) / (df['high'].shift(10) - df['low'].shift(10))
    
    # Efficiency divergence
    df['eff_div'] = df['eff_5d'] - df['eff_10d']
    
    # Volume-Price Confirmation Analysis
    # Daily price position
    df['price_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # 5-day volume-weighted price position
    df['volume_weighted_pos'] = (
        df['price_pos'].rolling(window=5).apply(lambda x: np.average(x, weights=df['volume'].loc[x.index]), raw=False)
    )
    
    # Volume-price divergence
    df['vol_price_div'] = df['eff_div'] * df['volume_weighted_pos']
    
    # Intraday Strength Persistence
    # Daily direction consistency
    df['daily_dir'] = np.sign(df['close'] - df['open']) * np.sign(df['close'].shift(1) - df['open'].shift(1))
    
    # 3-day sum of direction consistency
    df['persistence_3d'] = df['daily_dir'].rolling(window=3).sum()
    
    # Combine with volume-price signal
    df['combined_signal'] = df['vol_price_div'] * df['persistence_3d']
    
    # Adaptive Signal Generation
    # Apply asymmetric reversal logic
    def asymmetric_transform(x):
        if x > 0:
            return np.sqrt(x)
        else:
            return np.cbrt(x)
    
    df['asymmetric_signal'] = df['combined_signal'].apply(asymmetric_transform)
    
    # Scale by recent volume intensity
    df['volume_intensity'] = df['volume'] / df['volume'].shift(1)
    df['final_signal'] = df['asymmetric_signal'] * df['volume_intensity']
    
    return df['final_signal']
