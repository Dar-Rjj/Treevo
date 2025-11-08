import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Convergence Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Movement Framework
    # Momentum Components
    data['momentum_1d'] = data['close'] - data['close'].shift(1)
    data['intraday_momentum'] = data['close'] - data['open']
    
    data['momentum_3d'] = data['close'] - data['close'].shift(2)
    data['avg_momentum_3d'] = data['momentum_3d'] / 3
    
    data['momentum_5d'] = data['close'] - data['close'].shift(4)
    data['avg_momentum_5d'] = data['momentum_5d'] / 5
    
    # Volatility Context
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_3d'] = data['daily_range'].rolling(window=3).mean()
    data['avg_range_5d'] = data['daily_range'].rolling(window=5).mean()
    
    data['volatility_ratio'] = data['avg_range_3d'] / data['avg_range_5d']
    
    # Volatility Regime Classification
    data['vol_regime'] = 'normal'
    data.loc[data['volatility_ratio'] > 1.1, 'vol_regime'] = 'high'
    data.loc[data['volatility_ratio'] < 0.9, 'vol_regime'] = 'low'
    
    # Volume Analysis
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    data['volume_direction'] = np.sign(data['volume_change'])
    
    # Volume Streak calculation
    data['volume_streak'] = 0
    streak = 0
    for i in range(1, len(data)):
        if data['volume_direction'].iloc[i] == data['volume_direction'].iloc[i-1]:
            streak += 1
        else:
            streak = 1
        data['volume_streak'].iloc[i] = streak
    
    # Volume-Momentum Alignment
    data['direction_alignment'] = np.sign(data['momentum_1d']) * np.sign(data['volume_change'])
    
    # Alignment Streak calculation
    data['alignment_streak'] = 0
    align_streak = 0
    for i in range(1, len(data)):
        if data['direction_alignment'].iloc[i] > 0:
            align_streak += 1
        else:
            align_streak = 0
        data['alignment_streak'].iloc[i] = align_streak
    
    data['strength_alignment'] = np.abs(data['momentum_1d']) * np.abs(data['volume_change'])
    
    # Volume Regime Detection
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['vol_regime_volume'] = 'normal'
    data.loc[data['volume_ratio'] > 1.2, 'vol_regime_volume'] = 'high'
    data.loc[data['volume_ratio'] < 0.8, 'vol_regime_volume'] = 'low'
    
    # Factor Construction
    # Core Momentum Signal
    data['VSM_1d'] = data['momentum_1d'] / data['daily_range']
    data['VSM_3d'] = data['momentum_3d'] / data['avg_range_3d']
    data['VSM_5d'] = data['momentum_5d'] / data['avg_range_5d']
    
    # Multi-Timeframe Blend
    data['weighted_VSM'] = (5 * data['VSM_1d'] + 3 * data['VSM_3d'] + 2 * data['VSM_5d']) / 10
    data['core_momentum'] = data['weighted_VSM'] * np.log(data['volume'] + 1)
    
    # Volume Confirmation
    data['volume_persistence_base'] = data['volume_streak'] * np.abs(data['volume_change'])
    data['volume_persistence_scaled'] = data['volume_persistence_base'] / data['volume']
    
    data['alignment_multiplier'] = 1 + (data['alignment_streak'] * 0.1)
    data['strength_multiplier'] = 1 + (data['strength_alignment'] / data['volume'] * 0.05)
    
    data['volume_confirmation'] = data['volume_persistence_scaled'] * data['alignment_multiplier'] * data['strength_multiplier']
    
    # Regime-Adaptive Weights
    # Volatility Regime Scaling
    data['volatility_scaling'] = 1.0
    data.loc[data['vol_regime'] == 'high', 'volatility_scaling'] = 1.5
    data.loc[data['vol_regime'] == 'low', 'volatility_scaling'] = 1.3
    
    # Volume Regime Scaling
    data['volume_scaling'] = 1.0
    data.loc[data['vol_regime_volume'] == 'high', 'volume_scaling'] = 1.4
    data.loc[data['vol_regime_volume'] == 'low', 'volume_scaling'] = 0.6
    
    # Momentum Acceleration
    data['acceleration_signal'] = data['VSM_3d'] - data['VSM_5d']
    data['acceleration_direction'] = np.sign(data['acceleration_signal'])
    data['momentum_confirmation'] = 1 + (0.2 * data['acceleration_direction'])
    
    # Final Alpha Output
    # Composite Factor Value
    data['raw_factor'] = data['core_momentum'] * data['volume_confirmation']
    data['regime_adjusted'] = data['raw_factor'] * data['volatility_scaling'] * data['volume_scaling']
    data['final_alpha'] = data['regime_adjusted'] * data['momentum_confirmation']
    
    # Return the final alpha factor
    return data['final_alpha']
