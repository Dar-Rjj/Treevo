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
    
    # Core Price Components
    data['intraday_momentum'] = data['close'] - data['open']
    data['daily_range'] = data['high'] - data['low']
    data['overnight_gap'] = data['open'] - data['close'].shift(1)
    
    # Multi-Timeframe Momentum
    # Very Short-Term (1-2 days)
    data['price_momentum_vs'] = data['close'] - data['close'].shift(1)
    data['range_momentum_vs'] = data['daily_range'] - (data['high'].shift(1) - data['low'].shift(1))
    
    # Short-Term (3-5 days)
    data['price_momentum_st'] = data['close'] - data['close'].shift(4)
    data['range_momentum_st'] = data['daily_range'] - (data['high'].shift(4) - data['low'].shift(4))
    
    # Medium-Term (6-10 days)
    data['price_momentum_mt'] = data['close'] - data['close'].shift(9)
    data['range_momentum_mt'] = data['daily_range'] - (data['high'].shift(9) - data['low'].shift(9))
    
    # Volatility Regime Framework
    data['short_term_vol'] = data['daily_range'] + data['daily_range'].shift(1) + data['daily_range'].shift(2)
    data['medium_term_vol'] = data['daily_range'].rolling(window=10).sum()
    data['volatility_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Regime Classification
    data['vol_regime'] = 'normal'
    data.loc[data['volatility_ratio'] > 1.2, 'vol_regime'] = 'high'
    data.loc[data['volatility_ratio'] < 0.8, 'vol_regime'] = 'low'
    
    # Volume-Price Convergence
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    data['volume_direction'] = np.sign(data['volume_change'])
    
    # Volume Streak Calculation
    data['volume_streak'] = 0
    streak = 0
    for i in range(1, len(data)):
        if data['volume_direction'].iloc[i] == data['volume_direction'].iloc[i-1]:
            streak += 1
        else:
            streak = 0
        data.loc[data.index[i], 'volume_streak'] = streak
    
    # Price-Volume Alignment
    data['direction_alignment'] = np.sign(data['price_momentum_vs']) * data['volume_direction']
    
    # Alignment Streak Calculation
    data['alignment_streak'] = 0
    align_streak = 0
    for i in range(1, len(data)):
        if data['direction_alignment'].iloc[i] > 0:
            align_streak += 1
        else:
            align_streak = 0
        data.loc[data.index[i], 'alignment_streak'] = align_streak
    
    data['convergence_strength'] = data['alignment_streak'] * abs(data['price_momentum_vs'])
    
    # Volume Regime
    data['volume_3day'] = data['volume'] + data['volume'].shift(1) + data['volume'].shift(2)
    data['volume_10day_avg'] = data['volume'].rolling(window=10).mean() * 3
    data['volume_ratio'] = data['volume_3day'] / data['volume_10day_avg']
    
    data['volume_regime'] = 'normal'
    data.loc[data['volume_ratio'] > 1.1, 'volume_regime'] = 'high'
    data.loc[data['volume_ratio'] < 0.9, 'volume_regime'] = 'low'
    
    # Adaptive Factor Construction
    # Multi-Timeframe Blend with Regime-Adaptive Weights
    data['weighted_momentum'] = 0.0
    
    for idx in data.index:
        vol_regime = data.loc[idx, 'vol_regime']
        
        if vol_regime == 'high':
            weights = [0.5, 0.3, 0.2]  # vs, st, mt
        elif vol_regime == 'normal':
            weights = [0.3, 0.4, 0.3]  # vs, st, mt
        else:  # low volatility
            weights = [0.2, 0.3, 0.5]  # vs, st, mt
        
        momentum_vs = data.loc[idx, 'price_momentum_vs']
        momentum_st = data.loc[idx, 'price_momentum_st']
        momentum_mt = data.loc[idx, 'price_momentum_mt']
        
        data.loc[idx, 'weighted_momentum'] = (
            weights[0] * momentum_vs + 
            weights[1] * momentum_st + 
            weights[2] * momentum_mt
        )
    
    # Volume Integration
    data['volume_adjusted_momentum'] = data['weighted_momentum'] * np.log(data['volume'] + 1)
    data['convergence_boost'] = data['volume_adjusted_momentum'] * (1 + data['alignment_streak'] / 10)
    
    # Regime-Specific Adjustments
    data['volatility_scaling'] = 1.0
    data.loc[data['vol_regime'] == 'high', 'volatility_scaling'] = 0.7
    data.loc[data['vol_regime'] == 'low', 'volatility_scaling'] = 1.3
    
    data['volume_scaling'] = 1.0
    data.loc[data['volume_regime'] == 'high', 'volume_scaling'] = 1.2
    data.loc[data['volume_regime'] == 'low', 'volume_scaling'] = 0.8
    
    # Momentum Acceleration
    data['acceleration'] = (data['close'] - data['close'].shift(4)) - (data['close'].shift(4) - data['close'].shift(9))
    data['acceleration_direction'] = np.sign(data['acceleration'])
    data['momentum_confirmation'] = 1 + 0.1 * data['acceleration_direction']
    
    # Final Alpha Calculation
    data['raw_factor'] = data['convergence_boost']
    data['regime_adapted'] = data['raw_factor'] * data['volatility_scaling'] * data['volume_scaling']
    data['final_alpha'] = data['regime_adapted'] * data['momentum_confirmation']
    
    # Return the final alpha factor series
    return data['final_alpha']
