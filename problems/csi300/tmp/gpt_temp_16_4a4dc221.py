import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for volatility computation
    data['returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # Realized Volatility Components
    data['short_term_RV'] = data['returns'].rolling(window=5).apply(
        lambda x: np.sqrt(np.sum(x**2)), raw=False
    )
    data['medium_term_RV'] = data['returns'].rolling(window=10).apply(
        lambda x: np.sqrt(np.sum(x**2)), raw=False
    )
    data['volatility_ratio'] = data['short_term_RV'] / data['medium_term_RV']
    
    # Regime Classification
    conditions = [
        (data['volatility_ratio'] > 1.5) & (data['medium_term_RV'] > 0.02),
        (data['volatility_ratio'] < 0.7) & (data['medium_term_RV'] < 0.01)
    ]
    choices = ['high', 'low']
    data['regime'] = np.select(conditions, choices, default='normal')
    
    # Microstructure Anchoring System
    data['recent_high_anchor'] = data['high'].rolling(window=5).max()
    data['recent_low_anchor'] = data['low'].rolling(window=5).min()
    data['anchor_zone'] = data['recent_high_anchor'] - data['recent_low_anchor']
    
    # Volume-Based Anchoring
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['anchor_strength'] = (data['close'] - data['recent_low_anchor']) / data['anchor_zone']
    
    # Regime-Adaptive Momentum Construction
    data['high_vol_momentum'] = ((data['close'] - data['close'].shift(2)) / data['close'].shift(2)) * \
                               (1 + data['anchor_strength']) * data['volume_concentration']
    
    data['low_vol_momentum'] = ((data['close'] - data['close'].shift(9)) / data['close'].shift(9)) * \
                              (2 - abs(data['anchor_strength'] - 0.5)) * (1 + data['volume_concentration'])
    
    data['normal_vol_momentum'] = 0.3 * ((data['close'] - data['close'].shift(2)) / data['close'].shift(2)) + \
                                 0.7 * ((data['close'] - data['close'].shift(9)) / data['close'].shift(9))
    
    # Volatility-Adjusted Signal Enhancement
    data['volatility_dampening'] = 1 / (1 + data['medium_term_RV'])
    
    # Signal Persistence
    def calculate_persistence(series):
        if len(series) < 5:
            return np.nan
        current_sign = np.sign(series.iloc[-1])
        persistence_count = sum(np.sign(series.iloc[i]) == current_sign for i in range(len(series)))
        return persistence_count / 5
    
    data['returns_5d'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['signal_persistence'] = data['returns_5d'].rolling(window=5).apply(
        calculate_persistence, raw=False
    )
    
    # Final Factor Integration
    regime_mapping = {
        'high': data['high_vol_momentum'],
        'low': data['low_vol_momentum'],
        'normal': data['normal_vol_momentum']
    }
    
    data['regime_weighted_momentum'] = np.select(
        [data['regime'] == 'high', data['regime'] == 'low', data['regime'] == 'normal'],
        [data['high_vol_momentum'], data['low_vol_momentum'], data['normal_vol_momentum']]
    )
    
    data['enhanced_signal'] = data['regime_weighted_momentum'] * data['volatility_dampening'] * \
                            (1 + data['signal_persistence'])
    
    data['final_factor'] = data['enhanced_signal'] * data['anchor_strength']
    
    return data['final_factor']
