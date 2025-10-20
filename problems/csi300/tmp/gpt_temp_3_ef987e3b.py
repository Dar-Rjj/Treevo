import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price momentum acceleration hierarchy
    data['ultra_short_mom'] = data['close'].pct_change(3) - data['close'].pct_change(5)
    data['short_term_mom'] = data['close'].pct_change(5) - data['close'].pct_change(10)
    data['medium_term_mom'] = data['close'].pct_change(10) - data['close'].pct_change(20)
    
    # Momentum regime classification
    data['accelerating_momentum'] = ((data['ultra_short_mom'] > 0) & 
                                    (data['short_term_mom'] > 0) & 
                                    (data['medium_term_mom'] > 0)).astype(int)
    data['decelerating_momentum'] = ((data['ultra_short_mom'] < 0) & 
                                    (data['short_term_mom'] > 0) & 
                                    (data['medium_term_mom'] > 0)).astype(int)
    data['accelerating_reversal'] = ((data['ultra_short_mom'] < 0) & 
                                    (data['short_term_mom'] < 0) & 
                                    (data['medium_term_mom'] < 0)).astype(int)
    data['decelerating_reversal'] = ((data['ultra_short_mom'] > 0) & 
                                    (data['short_term_mom'] < 0) & 
                                    (data['medium_term_mom'] < 0)).astype(int)
    
    # Volume momentum acceleration structure
    data['ultra_short_vol_acc'] = (data['volume'].pct_change(3) - 
                                  data['volume'].pct_change(5))
    data['short_term_vol_acc'] = (data['volume'].pct_change(5) - 
                                 data['volume'].pct_change(10))
    data['medium_term_vol_acc'] = (data['volume'].pct_change(10) - 
                                  data['volume'].pct_change(20))
    
    # Volume regime classification
    data['volume_acceleration'] = ((data['ultra_short_vol_acc'] > 0) & 
                                  (data['short_term_vol_acc'] > 0) & 
                                  (data['medium_term_vol_acc'] > 0)).astype(int)
    data['volume_deceleration'] = ((data['ultra_short_vol_acc'] < 0) & 
                                  (data['short_term_vol_acc'] > 0) & 
                                  (data['medium_term_vol_acc'] > 0)).astype(int)
    
    # Range-based momentum metrics
    data['range_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['prev_close'] = data['close'].shift(1)
    data['gap_momentum'] = np.abs(data['open'] - data['prev_close']) / (data['high'] - data['low'] + 0.001)
    
    # Structure compression momentum
    data['daily_range'] = data['high'] - data['low']
    data['range_ma_20'] = data['daily_range'].rolling(window=20, min_periods=1).mean()
    data['structure_compression_momentum'] = 1 / ((data['daily_range'] / data['range_ma_20']) + 0.001)
    
    # Range efficiency persistence
    data['positive_range'] = (data['range_momentum'] > 0).astype(int)
    data['range_efficiency_persistence'] = data['positive_range'].rolling(window=5, min_periods=1).sum()
    
    # Price-volume-structure synchronization
    data['momentum_direction_alignment'] = (
        (np.sign(data['ultra_short_mom']) == np.sign(data['ultra_short_vol_acc'])).astype(int) +
        (np.sign(data['short_term_mom']) == np.sign(data['short_term_vol_acc'])).astype(int) +
        (np.sign(data['medium_term_mom']) == np.sign(data['medium_term_vol_acc'])).astype(int)
    )
    
    data['structure_momentum_convergence'] = data['range_momentum'] * data['ultra_short_mom']
    data['volume_momentum_confirmation'] = data['ultra_short_vol_acc'] * data['ultra_short_mom']
    
    # Multi-timeframe alignment strength
    mom_corr_data = pd.DataFrame({
        'ultra_short': data['ultra_short_mom'],
        'short_term': data['short_term_mom'],
        'medium_term': data['medium_term_mom']
    })
    data['multi_timeframe_alignment'] = mom_corr_data.rolling(window=10, min_periods=1).corr().groupby(level=0).mean().mean(axis=1)
    
    # Regime-Transition Acceleration Detection
    data['prev_accelerating_momentum'] = data['accelerating_momentum'].shift(1)
    data['prev_decelerating_momentum'] = data['decelerating_momentum'].shift(1)
    data['prev_volume_acceleration'] = data['volume_acceleration'].shift(1)
    data['prev_volume_deceleration'] = data['volume_deceleration'].shift(1)
    
    data['acceleration_initiation'] = ((data['prev_decelerating_momentum'] == 1) & 
                                      (data['accelerating_momentum'] == 1)).astype(int)
    data['volume_breakout'] = ((data['prev_volume_deceleration'] == 1) & 
                              (data['volume_acceleration'] == 1)).astype(int)
    
    # Compression release detection
    data['high_compression'] = (data['structure_compression_momentum'] > 
                               data['structure_compression_momentum'].rolling(window=10, min_periods=1).quantile(0.7)).astype(int)
    data['prev_high_compression'] = data['high_compression'].shift(1)
    data['compression_release'] = ((data['prev_high_compression'] == 1) & 
                                  (data['high_compression'] == 0) & 
                                  (data['range_momentum'] > 0)).astype(int)
    
    # Momentum quality scoring
    data['direction_consistency'] = (
        (data['ultra_short_mom'] > 0).astype(int) + 
        (data['short_term_mom'] > 0).astype(int) + 
        (data['medium_term_mom'] > 0).astype(int)
    ) / 3.0
    
    # Regime-based momentum multipliers
    data['regime_multiplier'] = 1.0
    data.loc[data['accelerating_momentum'] == 1, 'regime_multiplier'] = 1.5
    data.loc[data['decelerating_momentum'] == 1, 'regime_multiplier'] = 0.7
    data.loc[data['accelerating_reversal'] == 1, 'regime_multiplier'] = -1.2
    data.loc[data['decelerating_reversal'] == 1, 'regime_multiplier'] = -0.8
    
    # Volume-structure enhancement factors
    data['volume_confirmation_strength'] = np.tanh(data['volume_momentum_confirmation'])
    data['compression_release_intensity'] = data['compression_release'] * data['range_momentum']
    data['regime_change_intensity'] = (data['acceleration_initiation'] + data['volume_breakout']) * data['ultra_short_mom']
    
    # Base momentum construction
    data['base_momentum'] = (
        0.4 * data['ultra_short_mom'] + 
        0.35 * data['short_term_mom'] + 
        0.25 * data['medium_term_mom']
    )
    
    # Volume-enhanced momentum
    data['volume_enhanced_momentum'] = data['base_momentum'] * (1 + data['volume_confirmation_strength'])
    
    # Structure support quality
    data['structure_support_quality'] = data['range_momentum'] * data['range_efficiency_persistence']
    
    # Structure-optimized momentum
    data['structure_optimized_momentum'] = data['volume_enhanced_momentum'] * (1 + np.tanh(data['structure_support_quality']))
    
    # Asymmetric momentum dynamics
    data['upside_momentum'] = data['base_momentum'].clip(lower=0)
    data['downside_momentum'] = -data['base_momentum'].clip(upper=0)
    data['directional_bias'] = data['upside_momentum'] - data['downside_momentum']
    
    # Enhanced momentum signal
    data['enhanced_momentum_signal'] = data['structure_optimized_momentum'] * data['regime_multiplier']
    
    # Volume-timed prediction
    data['volume_timed_signal'] = data['enhanced_momentum_signal'] * (1 + data['volume_confirmation_strength'])
    
    # Structure-confirmed alpha
    data['structure_confirmed_alpha'] = data['volume_timed_signal'] * (1 + np.tanh(data['structure_support_quality']))
    
    # Asymmetric momentum alpha
    data['asymmetric_momentum_alpha'] = data['structure_confirmed_alpha'] * (1 + np.tanh(data['directional_bias']))
    
    # Regime-adaptive final alpha
    data['transition_bonus'] = data['regime_change_intensity'] * data['multi_timeframe_alignment']
    data['final_alpha'] = data['asymmetric_momentum_alpha'] * (1 + np.tanh(data['transition_bonus']))
    
    # Clean up intermediate columns
    result = data['final_alpha'].copy()
    
    return result
