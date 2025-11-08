import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Scaled Momentum with Volume Convergence alpha factor
    """
    # Core Price Components
    df['intraday_momentum'] = df['close'] - df['open']
    df['daily_range'] = df['high'] - df['low']
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Multi-Timeframe Momentum
    df['momentum_3d'] = df['close'] - df['close'].shift(2)
    df['range_3d'] = (df['high'] - df['low']) + (df['high'].shift(1) - df['low'].shift(1)) + (df['high'].shift(2) - df['low'].shift(2))
    
    df['momentum_10d'] = df['close'] - df['close'].shift(9)
    df['range_10d'] = df['high'].rolling(window=10).apply(lambda x: (x - df.loc[x.index, 'low']).sum(), raw=False)
    
    # Volatility Scaling
    df['vol_short'] = df['range_3d'] / 3
    df['vol_medium'] = df['range_10d'] / 10
    
    df['vsm_3d'] = df['momentum_3d'] / df['range_3d'].replace(0, np.nan)
    df['vsm_10d'] = df['momentum_10d'] / df['range_10d'].replace(0, np.nan)
    
    df['vol_ratio'] = df['vol_short'] / df['vol_medium']
    df['vol_regime'] = np.select(
        [df['vol_ratio'] > 1.1, df['vol_ratio'] < 0.9],
        ['high', 'low'],
        default='normal'
    )
    
    # Volume Analysis
    df['volume_direction'] = np.sign(df['volume'] - df['volume'].shift(1))
    
    # Volume streak calculation
    volume_streak = []
    current_streak = 0
    current_direction = 0
    
    for i, direction in enumerate(df['volume_direction']):
        if i == 0 or direction == 0:
            current_streak = 0
            current_direction = 0
        elif direction == current_direction:
            current_streak += 1
        else:
            current_streak = 1
            current_direction = direction
        
        volume_streak.append(current_streak)
    
    df['volume_streak'] = volume_streak
    df['volume_strength'] = df['volume_streak'] * abs(df['volume'] - df['volume'].shift(1))
    
    # Volume-Price Convergence
    df['direction_alignment'] = np.sign(df['price_change']) * np.sign(df['volume'] - df['volume'].shift(1))
    
    # Convergence streak calculation
    convergence_streak = []
    current_convergence = 0
    
    for i, alignment in enumerate(df['direction_alignment']):
        if i == 0 or alignment <= 0:
            current_convergence = 0
        else:
            current_convergence += 1
        
        convergence_streak.append(current_convergence)
    
    df['convergence_streak'] = convergence_streak
    df['convergence_strength'] = df['convergence_streak'] * abs(df['price_change'])
    
    # Factor Construction
    df['weighted_vsm'] = (2 * df['vsm_3d'] + df['vsm_10d']) / 3
    df['volume_integrated'] = df['weighted_vsm'] * np.log(df['volume'] + 1)
    
    # Volume Convergence Enhancement
    df['base_convergence'] = df['volume_integrated'] * (1 + df['convergence_streak'] / 5)
    df['base_volume_strength'] = df['volume_integrated'] * (1 + df['volume_strength'] / 1000)
    
    # Combined base signal
    df['base_signal'] = (df['base_convergence'] + df['base_volume_strength']) / 2
    
    # Volatility Adaptation
    df['final_alpha'] = df['base_signal'] * np.select(
        [df['vol_regime'] == 'high', df['vol_regime'] == 'low'],
        [0.7, 1.3],
        default=1.0
    )
    
    return df['final_alpha']
