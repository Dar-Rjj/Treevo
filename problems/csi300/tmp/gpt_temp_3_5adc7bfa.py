import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Multi-Scale Acceleration Divergence Analysis
    # Fractal Price Acceleration
    df['accel_3d'] = df['close'] / df['close'].shift(3) - 1
    df['accel_5d'] = df['close'] / df['close'].shift(5) - 1
    df['accel_10d'] = df['close'] / df['close'].shift(10) - 1
    df['accel_15d'] = df['close'] / df['close'].shift(15) - 1
    
    # Acceleration divergence sequences
    df['short_accel_div'] = (df['close'] / df['close'].shift(3) - df['close'] / df['close'].shift(5)).abs()
    df['medium_accel_div'] = (df['close'] / df['close'].shift(5) - df['close'] / df['close'].shift(10)).abs()
    df['long_accel_div'] = (df['close'] / df['close'].shift(10) - df['close'] / df['close'].shift(15)).abs()
    
    # Acceleration asymmetry patterns
    df['upside_accel_intensity'] = np.where(df['accel_3d'] > 0, df['accel_3d'], 0) + \
                                  np.where(df['accel_5d'] > 0, df['accel_5d'], 0) + \
                                  np.where(df['accel_10d'] > 0, df['accel_10d'], 0)
    
    df['downside_accel_intensity'] = np.where(df['accel_3d'] < 0, df['accel_3d'], 0) + \
                                    np.where(df['accel_5d'] < 0, df['accel_5d'], 0) + \
                                    np.where(df['accel_10d'] < 0, df['accel_10d'], 0)
    
    # Volume Fractal Acceleration Dynamics
    # Multi-scale volume momentum
    df['vol_momentum_3d'] = df['volume'].rolling(window=3).mean()
    df['vol_momentum_5d'] = df['volume'].rolling(window=5).mean()
    df['vol_momentum_10d'] = df['volume'].rolling(window=10).mean()
    
    # Volume acceleration alignment
    df['up_day_vol_accel'] = np.where(df['close'] > df['close'].shift(1), 
                                     df['volume'] / df['volume'].shift(3), 0)
    df['down_day_vol_accel'] = np.where(df['close'] < df['close'].shift(1), 
                                       df['volume'] / df['volume'].shift(3), 0)
    
    # Hierarchical Acceleration-Volume Integration
    # Volume ratios
    df['vol_ratio_short'] = df['vol_momentum_5d'] / df['vol_momentum_3d']
    df['vol_ratio_medium'] = df['vol_momentum_10d'] / df['vol_momentum_5d']
    
    # Directional acceleration pressure
    df['upward_accel_pressure'] = np.where(df['accel_3d'] > 0, df['accel_3d'] * df['vol_ratio_short'], 0)
    df['downward_accel_exhaustion'] = np.where(df['accel_3d'] < 0, df['accel_3d'] * df['vol_ratio_short'], 0)
    
    # Fractal Acceleration Regime Detection
    # Acceleration volatility ratios
    df['accel_vol_3d'] = df['accel_3d'].rolling(window=10).std()
    df['accel_vol_5d'] = df['accel_5d'].rolling(window=10).std()
    
    # Acceleration-volume efficiency
    df['accel_vol_efficiency'] = df['accel_3d'] / (df['vol_momentum_3d'] + 1e-8)
    
    # Hierarchical Factor Construction
    # Multi-dimensional acceleration signal combination
    df['short_term_factor'] = df['short_accel_div'] * df['vol_ratio_short']
    df['medium_term_factor'] = df['medium_accel_div'] * df['vol_ratio_medium']
    
    # Volume acceleration alignment for long-term factor
    df['vol_accel_alignment'] = (df['up_day_vol_accel'] - df['down_day_vol_accel']) / \
                               (df['up_day_vol_accel'] + df['down_day_vol_accel'] + 1e-8)
    df['long_term_factor'] = df['long_accel_div'] * df['vol_accel_alignment']
    
    # Dynamic Acceleration State Transitions
    # Acceleration superposition states
    df['accel_prob_short'] = df['accel_3d'].rolling(window=5).apply(lambda x: (x > 0).mean())
    df['accel_prob_medium'] = df['accel_5d'].rolling(window=5).apply(lambda x: (x > 0).mean())
    
    # State transition detection
    df['accel_jump_3d'] = (df['accel_3d'] - df['accel_3d'].shift(1)).abs()
    df['volume_breakpoint'] = (df['volume'] - df['vol_momentum_5d']).abs() / df['vol_momentum_5d']
    
    # Adaptive Signal Weighting
    # Regime-dependent acceleration strength
    high_vol_regime = df['accel_vol_3d'] > df['accel_vol_3d'].rolling(window=20).quantile(0.7)
    low_vol_regime = df['accel_vol_3d'] < df['accel_vol_3d'].rolling(window=20).quantile(0.3)
    
    df['regime_weight'] = np.where(high_vol_regime, 0.7, 
                                  np.where(low_vol_regime, 1.3, 1.0))
    
    # Final hierarchical acceleration factor
    df['hierarchical_accel_factor'] = (
        df['short_term_factor'] * 0.4 + 
        df['medium_term_factor'] * 0.35 + 
        df['long_term_factor'] * 0.25
    ) * df['regime_weight'] * df['accel_prob_short']
    
    # Clean up intermediate columns
    result = df['hierarchical_accel_factor'].copy()
    
    return result
