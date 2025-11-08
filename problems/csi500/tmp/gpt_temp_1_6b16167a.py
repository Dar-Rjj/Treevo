import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Core Momentum Calculation
    df['raw_momentum'] = df['close'] - df['open']
    df['direction'] = np.sign(df['raw_momentum'])
    df['daily_range'] = df['high'] - df['low']
    df['range_adjusted_momentum'] = df['raw_momentum'] / df['daily_range'].replace(0, np.nan)
    
    # Momentum Persistence Engine
    df['direction_change'] = df['direction'] != df['direction'].shift(1)
    persistence_count = 0
    persistence_scores = []
    
    for i, row in df.iterrows():
        if i == df.index[0]:
            persistence_count = 1
        else:
            if not df.loc[i, 'direction_change']:
                persistence_count += 1
            else:
                persistence_count = 1
        persistence_scores.append(persistence_count)
    
    df['persistence_count'] = persistence_scores
    df['persistence_weighted'] = df['persistence_count'] * df['range_adjusted_momentum'] * df['direction']
    
    # Volume Confirmation System
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['volume_direction'] = np.sign(df['volume'] - df['volume'].shift(1))
    df['price_volume_alignment'] = df['direction'] == df['volume_direction']
    df['alignment_strength'] = np.abs(df['volume_ratio'] - 1)
    df['confirmation_multiplier'] = df['alignment_strength'] * df['price_volume_alignment']
    
    # Volatility Decay Framework
    df['avg_5d_range'] = df['daily_range'].rolling(window=5, min_periods=1).mean()
    df['volatility_ratio'] = df['daily_range'] / df['avg_5d_range'].replace(0, np.nan)
    
    # Apply volatility-dependent decay to persistence
    decay_factor = 1 / (1 + 0.1 * df['volatility_ratio'])
    df['decayed_persistence'] = df['persistence_weighted'] * decay_factor
    
    # Final Alpha Integration
    df['alpha_raw'] = df['decayed_persistence'] * df['confirmation_multiplier'] * df['volatility_ratio']
    
    # Volatility-dependent smoothing
    smooth_window = np.where(df['volatility_ratio'] > 1.5, 3, 5)
    alpha_final = []
    
    for i in range(len(df)):
        window_size = int(smooth_window[i])
        start_idx = max(0, i - window_size + 1)
        alpha_final.append(df['alpha_raw'].iloc[start_idx:i+1].mean())
    
    result = pd.Series(alpha_final, index=df.index, name='alpha_factor')
    return result
