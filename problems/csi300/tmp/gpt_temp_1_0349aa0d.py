import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Price Behavior
    # Fractal Range Dynamics
    data['daily_range_fractal'] = (data['high'] - data['low']) / (abs(data['close'] - data['close'].shift(1)) + 1e-8)
    data['gap_absorption'] = abs(data['open'] - data['close'].shift(1)) / ((data['high'] - data['low']) + 1e-8)
    
    # 5-day average range
    data['avg_range_5d'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['range_persistence'] = ((data['high'] - data['low']) > data['avg_range_5d']).rolling(window=5).sum()
    
    # Price Position Asymmetry
    data['open_high_low_capture'] = (data['high'] - data['open']) / ((data['open'] - data['low']) + 1e-8)
    data['close_position'] = (data['close'] - data['low']) / ((data['high'] - data['low']) + 1e-8)
    
    max_open_close = np.maximum(data['open'], data['close'])
    min_open_close = np.minimum(data['open'], data['close'])
    data['net_rejection'] = ((data['high'] - max_open_close) - (min_open_close - data['low'])) / ((data['high'] - data['low']) + 1e-8)
    
    # Temporal Patterns
    data['morning_asymmetry'] = abs((data['high'] + data['low']) / 2 - data['open']) / ((data['high'] - data['low']) + 1e-8)
    data['session_bias'] = np.sign(data['close'] - data['open']) * np.sign((data['high'] + data['low']) / 2 - data['open'])
    
    # Volume Dynamics
    # Volume Acceleration
    data['volume_growth'] = (data['volume'] / (data['volume'].shift(3) + 1e-8)) ** (1/3) - 1
    data['amount_velocity'] = data['amount'] / (data['amount'].shift(1) + 1e-8) - 1
    
    # Volume-Price Interaction
    data['gap_volume_efficiency'] = data['volume'] / (abs(data['open'] - data['close'].shift(1)) + 1e-8)
    data['rejection_volume'] = data['volume'] * data['net_rejection']
    
    # Volume Efficiency
    true_range = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    data['high_efficiency'] = data['amount'] / ((data['volume'] * true_range) + 1e-8)
    
    # Efficiency Premium (simplified as deviation from mean)
    data['efficiency_premium'] = data['high_efficiency'] - data['high_efficiency'].rolling(window=5).mean()
    
    # Momentum Integration
    # Multi-Timeframe Rejection
    data['high_3d'] = data['close'].rolling(window=3).max()
    data['low_3d'] = data['close'].rolling(window=3).min()
    data['rejection_3d'] = (data['high'] - data['high_3d']) - (data['low_3d'] - data['low'])
    
    data['high_10d'] = data['close'].rolling(window=10).max()
    data['low_10d'] = data['close'].rolling(window=10).min()
    data['rejection_10d'] = (data['high'] - data['high_10d']) - (data['low_10d'] - data['low'])
    
    # Volume-Momentum Alignment
    data['volume_ma_3d'] = data['volume'].rolling(window=3).mean()
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_trend'] = np.sign(data['volume_ma_3d'] - data['volume_ma_5d'])
    data['range_momentum'] = np.sign(data['close'] - data['close'].shift(1)) * data['volume_trend']
    
    # Momentum Decay
    close_diff = np.sign(data['close'] - data['close'].shift(1))
    persistence = close_diff.groupby((close_diff != close_diff.shift(1)).cumsum()).cumcount() + 1
    data['persistence'] = persistence
    data['decay_adjusted_momentum'] = (data['close'] - data['close'].shift(1)) / (1 + data['persistence'])
    
    # Adaptive Weighting
    # Fractal Regime
    short_term_fractal = data['daily_range_fractal'].rolling(window=3).mean()
    medium_term_fractal = data['daily_range_fractal'].rolling(window=10).mean()
    data['high_fractal'] = short_term_fractal > medium_term_fractal
    data['low_fractal'] = short_term_fractal < medium_term_fractal
    
    # Regime Weights
    data['high_weight'] = data['range_persistence'] / 5
    
    # Range autocorrelation
    range_autocorr = (data['high'] - data['low']).rolling(window=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    data['low_weight'] = 1 - abs(range_autocorr)
    
    # Volume Scaling
    data['volume_weight'] = abs(data['volume_growth'])
    data['amount_weight'] = abs(data['amount_velocity'])
    
    # Composite Factor
    # Core Components
    regime_weight = np.where(data['high_fractal'], data['high_weight'], data['low_weight'])
    data['primary'] = (data['net_rejection'] * data['volume_growth']) * regime_weight
    data['secondary'] = data['range_momentum'] * data['session_bias']
    data['tertiary'] = data['efficiency_premium'] * data['morning_asymmetry']
    
    # Confirmation Layer
    data['volume_confirmation'] = data['amount_velocity'] * data['range_momentum']
    data['fractal_momentum'] = data['rejection_3d'] * data['decay_adjusted_momentum']
    
    # Final Factor
    # Regime-Weighted
    regime_weighted = (data['primary'] + data['secondary'] + data['tertiary'] + 
                      data['volume_confirmation'] + data['fractal_momentum']) * regime_weight
    
    # Volume-Scaled
    volume_scaling = (data['volume_weight'] + data['amount_weight']) / 2
    final_factor = regime_weighted * volume_scaling
    
    return final_factor
