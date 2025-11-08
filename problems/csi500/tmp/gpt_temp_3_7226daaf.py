import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum-Volume Convergence factor
    """
    df = data.copy()
    
    # Daily Returns
    df['intraday_return'] = df['close'] - df['open']
    df['overnight_return'] = df['open'] - df['close'].shift(1)
    
    # Price Range
    df['daily_range'] = df['high'] - df['low']
    df['gap'] = abs(df['open'] - df['close'].shift(1))
    
    # Volume Data
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Multi-Timeframe Momentum
    df['momentum_1d'] = df['close'] - df['close'].shift(1)
    df['momentum_3d'] = df['close'] - df['close'].shift(2)
    df['range_3d'] = (df['high'] - df['low']) + (df['high'].shift(1) - df['low'].shift(1)) + (df['high'].shift(2) - df['low'].shift(2))
    
    df['momentum_10d'] = df['close'] - df['close'].shift(9)
    df['range_10d'] = df['high'].rolling(window=10).apply(lambda x: (x - df.loc[x.index, 'low']).sum(), raw=False)
    
    # Volatility Regime Detection
    df['short_term_vol'] = df['range_3d'] / 3
    df['medium_term_vol'] = df['range_10d'] / 10
    df['volatility_ratio'] = df['short_term_vol'] / df['medium_term_vol']
    
    # Regime Classification
    df['volatility_regime'] = 'normal'
    df.loc[df['volatility_ratio'] > 1.2, 'volatility_regime'] = 'high'
    df.loc[df['volatility_ratio'] < 0.8, 'volatility_regime'] = 'low'
    
    # Volume Persistence Analysis
    df['volume_direction'] = np.sign(df['volume_change'])
    
    # Volume streak calculation
    volume_streak = []
    current_streak = 1
    current_direction = 0
    
    for i, direction in enumerate(df['volume_direction']):
        if i == 0 or direction == 0:
            volume_streak.append(1)
            current_direction = direction
        elif direction == current_direction:
            current_streak += 1
            volume_streak.append(current_streak)
        else:
            current_streak = 1
            volume_streak.append(current_streak)
            current_direction = direction
    
    df['volume_streak'] = volume_streak
    
    # Volume-Momentum Alignment
    df['alignment'] = np.sign(df['momentum_1d']) * np.sign(df['volume_change'])
    
    # Alignment streak calculation
    alignment_streak = []
    current_alignment_streak = 1
    current_alignment = 0
    
    for i, alignment in enumerate(df['alignment']):
        if i == 0 or alignment <= 0:
            alignment_streak.append(1)
            current_alignment = alignment
        elif alignment == current_alignment:
            current_alignment_streak += 1
            alignment_streak.append(current_alignment_streak)
        else:
            current_alignment_streak = 1
            alignment_streak.append(current_alignment_streak)
            current_alignment = alignment
    
    df['alignment_streak'] = alignment_streak
    
    # Volume Regime
    df['volume_3d'] = df['volume'].rolling(window=3).mean()
    df['volume_10d'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume_3d'] / df['volume_10d']
    
    df['volume_regime'] = 'normal'
    df.loc[df['volume_ratio'] > 1.1, 'volume_regime'] = 'high'
    df.loc[df['volume_ratio'] < 0.9, 'volume_regime'] = 'low'
    
    # Core Factor Construction
    # Base Momentum Signal
    df['VSM_3d'] = df['momentum_3d'] / df['range_3d']
    df['VSM_10d'] = df['momentum_10d'] / df['range_10d']
    df['momentum_blend'] = (2 * df['VSM_3d'] + df['VSM_10d']) / 3
    
    # Volume Confirmation
    df['volume_persistence'] = df['volume_streak'] * abs(df['volume_change'])
    df['alignment_strength'] = df['alignment_streak'] * abs(df['momentum_1d'])
    df['volume_score'] = (df['volume_persistence'] + df['alignment_strength']) / 2
    
    # Regime-Adaptive Weights
    df['volatility_weight'] = 0.5  # default normal
    df.loc[df['volatility_regime'] == 'high', 'volatility_weight'] = 0.7
    df.loc[df['volatility_regime'] == 'low', 'volatility_weight'] = 0.3
    
    df['volume_weight'] = 0.3  # default normal
    df.loc[df['volume_regime'] == 'high', 'volume_weight'] = 0.4
    df.loc[df['volume_regime'] == 'low', 'volume_weight'] = 0.2
    
    # Final Alpha Calculation
    df['volatility_adjusted_momentum'] = df['volatility_weight'] * df['momentum_blend']
    df['volume_enhanced_momentum'] = df['volume_weight'] * df['volume_score']
    df['raw_alpha'] = df['volatility_adjusted_momentum'] + df['volume_enhanced_momentum']
    
    # Clean up and return
    alpha_series = df['raw_alpha'].copy()
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan)
    
    return alpha_series
