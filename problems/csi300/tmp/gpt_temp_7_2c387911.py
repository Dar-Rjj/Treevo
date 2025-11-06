import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Identification
    # Realized Volatility Estimation
    data['daily_range_vol'] = (data['high'] - data['low']) / data['close']
    data['close_to_close_vol'] = abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['garman_klass'] = np.sqrt(
        0.5 * (np.log(data['high'] / data['low']))**2 - 
        (2*np.log(2)-1) * (np.log(data['close'] / data['open']))**2
    )
    
    # Regime Classification
    data['daily_range_vol_10d_avg'] = data['daily_range_vol'].rolling(window=10, min_periods=1).mean()
    data['vol_regime'] = 'normal'
    data.loc[data['daily_range_vol'] > 1.5 * data['daily_range_vol_10d_avg'], 'vol_regime'] = 'high'
    data.loc[data['daily_range_vol'] < 0.7 * data['daily_range_vol_10d_avg'], 'vol_regime'] = 'low'
    
    # Volatility Momentum
    data['vol_acceleration'] = data['daily_range_vol'] / data['daily_range_vol'].shift(1)
    data['vol_mean_reversion'] = 1 / data['vol_acceleration']
    
    # Volatility Regime Persistence
    regime_persistence = []
    for i in range(len(data)):
        if i < 4:
            regime_persistence.append(0.2)
        else:
            current_regime = data['vol_regime'].iloc[i]
            same_count = sum(data['vol_regime'].iloc[i-4:i] == current_regime)
            regime_persistence.append(same_count / 5)
    data['vol_regime_persistence'] = regime_persistence
    
    # Microstructure Anchoring Framework
    # Price Level Anchors
    data['opening_anchor_strength'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['prev_close_anchor'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['intraday_hl_anchors'] = (data['close'] - data['low']) / (data['high'] - data['low']) - 0.5
    
    # Anchor Persistence Metrics
    data['opening_anchor_persistence'] = np.sign(data['close'] - data['open']) * np.sign(data['close'].shift(1) - data['open'].shift(1))
    data['close_anchor_persistence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['close'].shift(1) - data['close'].shift(2))
    data['range_anchor_stability'] = 1 - abs((data['close'] - data['low']) / (data['high'] - data['low']) - 0.5)
    
    # Anchor Breakout Signals
    data['opening_breakout'] = abs(data['close'] - data['open']) / (0.5 * (data['high'] - data['low']))
    data['prev_close_breakout'] = abs(data['close'] - data['close'].shift(1)) / (0.5 * (data['high'].shift(1) - data['low'].shift(1)))
    data['range_breakout'] = np.maximum(data['high'] - data['high'].shift(1), data['low'].shift(1) - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Regime-Adaptive Momentum Construction
    # Short-term Momentum Components
    data['raw_price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['range_adjusted_momentum'] = (data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['volume_confirmed_momentum'] = data['raw_price_momentum'] * (data['volume'] / data['volume'].shift(1))
    
    # Medium-term Momentum Filters
    momentum_consistency = []
    for i in range(len(data)):
        if i < 4:
            momentum_consistency.append(0.25)
        else:
            current_sign = np.sign(data['close'].iloc[i] - data['close'].iloc[i-1])
            same_count = sum(np.sign(data['close'].iloc[i-4:i] - data['close'].iloc[i-5:i-1]) == current_sign)
            momentum_consistency.append(same_count / 4)
    data['momentum_5d_consistency'] = momentum_consistency
    
    data['momentum_acceleration'] = (data['close'] / data['close'].shift(1)) / (data['close'].shift(1) / data['close'].shift(2))
    data['vol_adjusted_momentum'] = data['raw_price_momentum'] / data['daily_range_vol']
    
    # Momentum Regime Adaptation
    data['high_vol_momentum'] = data['range_adjusted_momentum'] * data['vol_acceleration']
    data['low_vol_momentum'] = data['raw_price_momentum'] * data['vol_mean_reversion']
    data['normal_vol_momentum'] = data['volume_confirmed_momentum'] * data['momentum_acceleration']
    
    # Regime-Weighted Momentum
    def get_regime_weighted_momentum(row):
        if row['vol_regime'] == 'high':
            return row['high_vol_momentum']
        elif row['vol_regime'] == 'low':
            return row['low_vol_momentum']
        else:
            return row['normal_vol_momentum']
    
    data['regime_weighted_momentum'] = data.apply(get_regime_weighted_momentum, axis=1)
    
    # Volume-Microstructure Integration
    # Volume Anchor Confirmation
    data['opening_volume_anchor'] = data['volume'] * data['opening_anchor_strength']
    data['close_volume_anchor'] = data['volume'] * abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['range_volume_density'] = data['volume'] / (data['high'] - data['low'])
    
    # Volume Regime Dynamics
    data['volume_vol_ratio'] = (data['volume'] / data['volume'].shift(1)) / (data['daily_range_vol'] / data['daily_range_vol'].shift(1))
    
    volume_persistence = []
    for i in range(len(data)):
        if i < 4:
            volume_persistence.append(0.25)
        else:
            current_sign = np.sign(data['volume'].iloc[i] - data['volume'].iloc[i-1])
            same_count = sum(np.sign(data['volume'].iloc[i-4:i] - data['volume'].iloc[i-5:i-1]) == current_sign)
            volume_persistence.append(same_count / 4)
    data['volume_regime_persistence'] = volume_persistence
    
    data['volume_momentum_alignment'] = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['close'] - data['close'].shift(1))
    
    # Microstructure Volume Signals
    data['anchor_breakout_volume'] = data['volume'] * data['opening_breakout']
    data['range_compression_volume'] = data['volume'] * (1 - data['daily_range_vol'] / data['daily_range_vol_10d_avg'])
    data['volume_anchor_divergence'] = data['opening_volume_anchor'] - data['close_volume_anchor']
    
    # Cross-Timeframe Validation
    # Multi-scale Anchor Consistency
    data['short_term_anchor_alignment'] = np.sign(data['opening_anchor_strength']) * np.sign(data['prev_close_anchor'])
    
    anchor_stability = []
    for i in range(len(data)):
        if i < 4:
            anchor_stability.append(0.25)
        else:
            current_sign = np.sign(data['close'].iloc[i] - data['open'].iloc[i])
            same_count = sum(np.sign(data['close'].iloc[i-4:i] - data['open'].iloc[i-4:i]) == current_sign)
            anchor_stability.append(same_count / 4)
    data['medium_term_anchor_stability'] = anchor_stability
    
    data['anchor_momentum_coherence'] = np.sign(data['opening_anchor_strength']) * np.sign(data['raw_price_momentum'])
    
    # Regime-Momentum Validation
    data['vol_momentum_fit'] = data['vol_adjusted_momentum'] * data['vol_regime_persistence']
    data['regime_transition_momentum'] = data['raw_price_momentum'] * (1 - data['vol_regime_persistence'])
    
    momentum_regime_consistency = []
    for i in range(len(data)):
        if i < 4:
            momentum_regime_consistency.append(0.2)
        else:
            regime_multiplier = 1 if data['vol_regime'].iloc[i] == 'high' else (-1 if data['vol_regime'].iloc[i] == 'low' else 0)
            same_count = sum(np.sign(data['raw_price_momentum'].iloc[i-4:i+1]) == regime_multiplier)
            momentum_regime_consistency.append(same_count / 5)
    data['momentum_regime_consistency'] = momentum_regime_consistency
    
    # Volume-Anchor Validation
    data['volume_anchor_persistence'] = data['volume_regime_persistence'] * data['opening_anchor_persistence']
    data['breakout_volume_confirmation'] = data['anchor_breakout_volume'] * data['volume_momentum_alignment']
    data['microstructure_volume_validation'] = data['range_volume_density'] * data['range_anchor_stability']
    
    # Adaptive Alpha Synthesis
    # Core Signal Components
    data['regime_adapted_momentum'] = data['regime_weighted_momentum'] * data['momentum_regime_consistency']
    data['volume_confirmed_anchors'] = data['volume_anchor_divergence'] * data['volume_anchor_persistence']
    data['volatility_enhanced_breakouts'] = data['opening_breakout'] * data['vol_acceleration']
    data['microstructure_momentum'] = data['range_adjusted_momentum'] * data['opening_anchor_persistence']
    
    # Regime-Specific Enhancement
    def apply_regime_multipliers(row):
        base_signals = {
            'regime_adapted_momentum': row['regime_adapted_momentum'],
            'volume_confirmed_anchors': row['volume_confirmed_anchors'],
            'volatility_enhanced_breakouts': row['volatility_enhanced_breakouts'],
            'microstructure_momentum': row['microstructure_momentum']
        }
        
        if row['vol_regime'] == 'high':
            base_signals['volatility_enhanced_breakouts'] *= 1.6
        elif row['vol_regime'] == 'low':
            base_signals['regime_adapted_momentum'] *= 0.8
        else:
            base_signals['volume_confirmed_anchors'] *= 1.2
        
        if row['vol_regime_persistence'] < 0.6:
            for key in base_signals:
                base_signals[key] *= 1.4
        
        return base_signals
    
    enhanced_signals = data.apply(apply_regime_multipliers, axis=1)
    for key in ['regime_adapted_momentum', 'volume_confirmed_anchors', 'volatility_enhanced_breakouts', 'microstructure_momentum']:
        data[f'{key}_enhanced'] = enhanced_signals.apply(lambda x: x[key])
    
    # Validation-Weighted Signals
    data['validated_regime_momentum'] = data['regime_adapted_momentum_enhanced'] * data['vol_momentum_fit']
    data['confirmed_anchor_breakouts'] = data['volatility_enhanced_breakouts_enhanced'] * data['breakout_volume_confirmation']
    data['persistent_microstructure'] = data['volume_confirmed_anchors_enhanced'] * data['medium_term_anchor_stability']
    data['aligned_momentum_anchors'] = data['microstructure_momentum_enhanced'] * data['anchor_momentum_coherence']
    
    # Final Alpha Construction
    primary_factor = data['validated_regime_momentum'] * data['volume_momentum_alignment']
    secondary_factor = data['confirmed_anchor_breakouts'] * data['range_anchor_stability']
    tertiary_factor = data['persistent_microstructure'] * data['volume_vol_ratio']
    quaternary_factor = data['aligned_momentum_anchors'] * data['momentum_acceleration']
    
    # Combine factors with weights
    alpha = (0.4 * primary_factor + 0.3 * secondary_factor + 
             0.2 * tertiary_factor + 0.1 * quaternary_factor)
    
    return alpha
