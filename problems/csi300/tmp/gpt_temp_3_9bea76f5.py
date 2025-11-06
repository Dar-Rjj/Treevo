import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Decay Volume-Weighted Fractal Breakout Factor
    Combines fractal momentum analysis, volume-weighted price convergence, 
    multi-timeframe breakout detection, and volume fractal consistency
    """
    data = df.copy()
    
    # Fractal Momentum Structure Analysis
    # Multi-Scale Momentum Calculation
    data['momentum_2d'] = data['close'] - data['close'].shift(1)
    data['momentum_5d'] = data['close'] - data['close'].shift(4)
    data['momentum_8d'] = data['close'] - data['close'].shift(7)
    
    # Momentum Decay Assessment
    data['momentum_decay_rate'] = data['momentum_2d'] / (data['momentum_5d'] + 1e-8)
    data['momentum_persistence'] = data['momentum_5d'] / (data['momentum_8d'] + 1e-8)
    data['fractal_momentum_score'] = data['momentum_decay_rate'] * data['momentum_persistence']
    
    # Volume-Weighted Price Fractal Analysis
    # Volume-Weighted Price Levels
    data['vwap_5'] = (data['close'] * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    data['vwap_10'] = (data['close'] * data['volume']).rolling(window=10).sum() / data['volume'].rolling(window=10).sum()
    data['vwap_20'] = (data['close'] * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
    
    # Fractal Price Convergence
    data['short_term_conv'] = abs(data['close'] - data['vwap_5']) / (data['vwap_5'] + 1e-8)
    data['medium_term_conv'] = abs(data['vwap_5'] - data['vwap_10']) / (data['vwap_10'] + 1e-8)
    data['long_term_conv'] = abs(data['vwap_10'] - data['vwap_20']) / (data['vwap_20'] + 1e-8)
    data['fractal_convergence_score'] = 1 - (data['short_term_conv'] + data['medium_term_conv'] + data['long_term_conv']) / 3
    
    # Breakout Pattern Recognition
    # Multi-Timeframe Breakout Detection
    data['high_5d'] = data['high'].rolling(window=4, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['low_5d'] = data['low'].rolling(window=4, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    data['breakout_5d'] = ((data['close'] > data['high_5d']) | (data['close'] < data['low_5d'])).astype(float)
    
    data['high_10d'] = data['high'].rolling(window=9, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['low_10d'] = data['low'].rolling(window=9, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    data['breakout_10d'] = ((data['close'] > data['high_10d']) | (data['close'] < data['low_10d'])).astype(float)
    
    data['high_20d'] = data['high'].rolling(window=19, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['low_20d'] = data['low'].rolling(window=19, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    data['breakout_20d'] = ((data['close'] > data['high_20d']) | (data['close'] < data['low_20d'])).astype(float)
    
    # Breakout Strength Assessment
    data['breakout_dist_5d'] = np.where(data['close'] > data['high_5d'], 
                                       (data['close'] - data['high_5d']) / (data['high_5d'] + 1e-8),
                                       (data['low_5d'] - data['close']) / (data['low_5d'] + 1e-8))
    data['breakout_dist_10d'] = np.where(data['close'] > data['high_10d'], 
                                        (data['close'] - data['high_10d']) / (data['high_10d'] + 1e-8),
                                        (data['low_10d'] - data['close']) / (data['low_10d'] + 1e-8))
    data['breakout_dist_20d'] = np.where(data['close'] > data['high_20d'], 
                                        (data['close'] - data['high_20d']) / (data['high_20d'] + 1e-8),
                                        (data['low_20d'] - data['close']) / (data['low_20d'] + 1e-8))
    
    # Composite Breakout Score
    weights = [0.5, 0.3, 0.2]
    data['composite_breakout_score'] = (data['breakout_dist_5d'] * weights[0] * data['breakout_5d'] + 
                                       data['breakout_dist_10d'] * weights[1] * data['breakout_10d'] + 
                                       data['breakout_dist_20d'] * weights[2] * data['breakout_20d'])
    
    # Volume Fractal Analysis
    # Volume Momentum Structure
    data['volume_change_2d'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    data['volume_change_5d'] = data['volume'] / (data['volume'].rolling(window=4, min_periods=1).apply(lambda x: x[:-1].mean() if len(x) > 1 else np.nan) + 1e-8)
    data['volume_change_10d'] = data['volume'] / (data['volume'].rolling(window=9, min_periods=1).apply(lambda x: x[:-1].mean() if len(x) > 1 else np.nan) + 1e-8)
    
    # Volume Fractal Consistency
    data['volume_momentum_ratio'] = data['volume_change_2d'] / (data['volume_change_5d'] + 1e-8)
    data['volume_persistence'] = data['volume_change_5d'] / (data['volume_change_10d'] + 1e-8)
    data['volume_fractal_score'] = data['volume_momentum_ratio'] * data['volume_persistence']
    
    # Adaptive Factor Integration
    # Core Factor Construction
    data['base_factor'] = data['fractal_momentum_score'] * data['fractal_convergence_score']
    
    # Breakout Enhancement
    data['breakout_enhanced_factor'] = data['base_factor'] * (1 + data['composite_breakout_score'])
    
    # Volume Confirmation
    volume_multiplier = np.where(data['volume_fractal_score'] > 1.2, 1.5,
                                np.where(data['volume_fractal_score'] > 1.0, 1.2, 1.0))
    data['volume_confirmed_factor'] = data['breakout_enhanced_factor'] * volume_multiplier
    
    # Momentum Decay Adjustment
    decay_adjustment = np.where(data['momentum_decay_rate'] > 1.5, 0.7,
                               np.where(data['momentum_decay_rate'] < 0.7, 1.2, 1.0))
    data['final_alpha'] = data['volume_confirmed_factor'] * decay_adjustment
    
    return data['final_alpha']
