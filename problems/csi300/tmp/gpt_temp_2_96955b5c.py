import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-timeframe price acceleration
    data['upside_accel'] = (data['high'] - data['high'].shift(1)) - (data['high'].shift(1) - data['high'].shift(2))
    data['downside_accel'] = (data['low'] - data['low'].shift(1)) - (data['low'].shift(1) - data['low'].shift(2))
    data['medium_term_price'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    data['long_term_price'] = (data['close'] - data['close'].shift(21)) / data['close'].shift(21)
    
    # Volume momentum patterns
    data['volume_accel'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_persistence'] = data['volume'].rolling(window=5).apply(
        lambda x: (x > x.median()).sum(), raw=False
    )
    data['short_vol_momentum'] = data['volume'] / data['volume'].shift(3)
    data['medium_vol_persistence'] = data['volume'] / data['volume'].shift(5)
    
    # Asymmetric divergence detection
    data['pos_divergence'] = ((data['upside_accel'] > 0) & (data['short_vol_momentum'] < 1)).astype(int)
    data['neg_divergence'] = ((data['downside_accel'] < 0) & (data['short_vol_momentum'] > 1)).astype(int)
    data['divergence_strength'] = abs(data['upside_accel']) * abs(data['short_vol_momentum'] - 1)
    
    # Gap dynamics
    data['daily_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_accel'] = data['daily_gap'] - data['daily_gap'].shift(1)
    data['gap_persistence'] = data['daily_gap'].rolling(window=3).apply(
        lambda x: 1 if (x > 0).all() else (-1 if (x < 0).all() else 0), raw=False
    )
    
    # Fractal efficiency metrics
    data['movement_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Fractal Efficiency: |Close_t - Close_{t-10}| / Sum(|Close_i - Close_{i-1}| from i=t-9 to t)
    price_changes = abs(data['close'].diff())
    data['fractal_efficiency'] = abs(data['close'] - data['close'].shift(10)) / (
        price_changes.rolling(window=10).sum() + 1e-8
    )
    data['range_price_reversal'] = data['close'].diff() / (data['high'] - data['low'] + 1e-8)
    
    # Volatility regime detection
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['short_term_tr'] = data['true_range'].rolling(window=3).mean()
    data['medium_term_tr'] = data['true_range'].rolling(window=8).mean()
    data['vol_compression'] = data['short_term_tr'] / (data['medium_term_tr'] + 1e-8)
    
    # Asymmetric momentum weighting
    epsilon = 1e-8
    data['volume_upside_momentum'] = data['upside_accel'] * data['short_vol_momentum']
    data['volume_downside_momentum'] = data['downside_accel'] * data['medium_vol_persistence']
    data['asymmetric_ratio'] = data['volume_upside_momentum'] / (abs(data['volume_downside_momentum']) + epsilon)
    
    # Gap-enhanced momentum
    data['gap_momentum_interaction'] = data['gap_accel'] * data['upside_accel']
    data['gap_volume_alignment'] = data['gap_persistence'] * data['volume_persistence']
    data['gap_fractal_efficiency'] = data['gap_accel'] * data['fractal_efficiency']
    
    # Multi-timeframe signal synthesis
    data['short_term_signal'] = data['volume_upside_momentum'] * data['gap_accel']
    data['medium_term_signal'] = data['asymmetric_ratio'] * data['volume_persistence']
    data['long_term_signal'] = data['gap_fractal_efficiency'] * (2 - data['vol_compression'])
    
    # Efficiency-regime filtering
    data['high_efficiency'] = ((abs(data['movement_efficiency']) > 0.5) & 
                              (data['fractal_efficiency'] > 0.6)).astype(int)
    data['low_efficiency'] = ((abs(data['movement_efficiency']) < 0.3) | 
                             (data['fractal_efficiency'] < 0.4)).astype(int)
    data['neutral_efficiency'] = ((~data['high_efficiency'].astype(bool)) & 
                                 (~data['low_efficiency'].astype(bool))).astype(int)
    
    # Regime-adaptive weighting
    data['high_eff_multiplier'] = 1 + data['movement_efficiency'] * data['fractal_efficiency']
    data['low_eff_penalty'] = 0.5
    data['volatility_adjustment'] = 2 - data['vol_compression']
    
    # Final factor integration
    data['base_signal'] = data['asymmetric_ratio'] * data['gap_momentum_interaction']
    data['volume_confirmed'] = data['base_signal'] * data['volume_persistence']
    data['fractal_enhanced'] = data['volume_confirmed'] * data['fractal_efficiency']
    
    # Multi-timeframe aggregation with weights
    short_weight, medium_weight, long_weight = 0.4, 0.35, 0.25
    data['raw_factor'] = (short_weight * data['short_term_signal'] + 
                         medium_weight * data['medium_term_signal'] + 
                         long_weight * data['long_term_signal'])
    
    # Efficiency adjustment
    data['efficiency_adjustment'] = np.where(
        data['high_efficiency'] == 1, data['high_eff_multiplier'],
        np.where(data['low_efficiency'] == 1, data['low_eff_penalty'], 1)
    )
    data['efficiency_adjusted'] = data['raw_factor'] * data['efficiency_adjustment'] * data['volatility_adjustment']
    
    # Final factor
    data['final_factor'] = data['efficiency_adjusted'] * data['gap_volume_alignment']
    
    return data['final_factor']
