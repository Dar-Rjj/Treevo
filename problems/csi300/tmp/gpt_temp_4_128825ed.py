import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Transition with Momentum Persistence alpha factor
    """
    data = df.copy()
    
    # Calculate basic price and volume metrics
    data['daily_range'] = data['high'] - data['low']
    data['prev_range'] = data['daily_range'].shift(1)
    data['range_ratio'] = data['daily_range'] / data['prev_range']
    data['daily_return'] = data['close'] / data['open'] - 1
    
    # Volatility Compression Patterns
    # Consecutive compression days (range < previous range)
    data['compression_day'] = (data['range_ratio'] < 1).astype(int)
    data['consecutive_compression'] = data['compression_day'] * (data['compression_day'].groupby(data.index).cumsum())
    
    # Volatility momentum - range expansion acceleration
    data['range_ma_5'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['range_expansion'] = data['daily_range'] / data['range_ma_5'].shift(1)
    data['range_change_rate'] = data['daily_range'] / data['daily_range'].shift(1) - 1
    
    # Breakout direction identification
    data['breakout_threshold'] = 0.7 * data['daily_range']
    data['upward_break'] = (data['close'] > data['open'] + data['breakout_threshold']).astype(int)
    data['downward_break'] = (data['close'] < data['open'] - data['breakout_threshold']).astype(int)
    
    # Volatility regime score components
    data['compression_intensity'] = data['consecutive_compression'] / 5  # normalized
    data['expansion_magnitude'] = np.tanh(data['range_expansion'] - 1)  # bounded
    data['directional_bias'] = data['upward_break'] - data['downward_break']
    
    # Volatility clustering analysis
    data['range_autocorr'] = data['daily_range'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Regime transition points
    data['compression_to_expansion'] = ((data['range_ratio'] > 1.2) & 
                                      (data['range_ratio'].shift(1) < 0.8)).astype(int)
    data['expansion_to_compression'] = ((data['range_ratio'] < 0.8) & 
                                      (data['range_ratio'].shift(1) > 1.2)).astype(int)
    
    # Cluster strength indicator
    data['regime_duration'] = data['compression_day'].rolling(window=10, min_periods=3).apply(
        lambda x: len(x) - np.argmax(x[::-1]) if np.any(x) else 0, raw=False
    )
    data['recent_change_magnitude'] = data['daily_range'].rolling(window=5, min_periods=3).std()
    data['volatility_consistency'] = 1 - (data['daily_range'].rolling(window=5, min_periods=3).std() / 
                                        data['daily_range'].rolling(window=5, min_periods=3).mean())
    
    # Combined volatility signals
    data['strong_momentum_init'] = ((data['compression_intensity'] > 0.6) & 
                                  (data['upward_break'] == 1)).astype(int)
    data['trend_persistence'] = ((data['expansion_magnitude'] > 0.3) & 
                               (data['directional_bias'].rolling(window=3, min_periods=2).mean() > 0)).astype(int)
    
    # Momentum quality assessment
    data['return_per_volume'] = (data['close'] - data['open']) / (data['volume'] + 1e-8)
    data['momentum_efficiency'] = data['return_per_volume'].rolling(window=5, min_periods=3).mean()
    
    # Momentum consistency
    data['direction_persistence'] = (data['daily_return'] > 0).rolling(window=5, min_periods=3).mean()
    data['magnitude_stability'] = 1 - (abs(data['daily_return']).rolling(window=5, min_periods=3).std() / 
                                     abs(data['daily_return']).rolling(window=5, min_periods=3).mean())
    
    # Momentum acceleration
    data['return_acceleration'] = data['daily_return'] / (abs(data['daily_return'].shift(1)) + 1e-8)
    data['momentum_accel_trend'] = data['return_acceleration'].rolling(window=3, min_periods=2).mean()
    
    # Momentum quality score
    data['efficiency_component'] = np.tanh(data['momentum_efficiency'] * 1000)
    data['consistency_component'] = (data['direction_persistence'] + data['magnitude_stability']) / 2
    data['acceleration_component'] = np.tanh(data['momentum_accel_trend'])
    data['momentum_quality'] = (data['efficiency_component'] + data['consistency_component'] + 
                              data['acceleration_component']) / 3
    
    # Volume-momentum relationship
    data['up_day_volume'] = data['volume'].where(data['daily_return'] > 0)
    data['down_day_volume'] = data['volume'].where(data['daily_return'] < 0)
    data['volume_support_up'] = (data['up_day_volume'].rolling(window=5, min_periods=3).mean() / 
                               data['down_day_volume'].rolling(window=5, min_periods=3).mean())
    data['volume_support_down'] = (data['down_day_volume'].rolling(window=5, min_periods=3).mean() / 
                                 data['up_day_volume'].rolling(window=5, min_periods=3).mean())
    
    # Volume divergences
    data['price_up_volume_down'] = ((data['daily_return'] > 0) & 
                                  (data['volume'] < data['volume'].rolling(window=5, min_periods=3).mean())).astype(int)
    data['price_down_volume_up'] = ((data['daily_return'] < 0) & 
                                  (data['volume'] > data['volume'].rolling(window=5, min_periods=3).mean())).astype(int)
    
    # Volume-momentum alignment
    data['confirmation_strength'] = np.where(data['daily_return'] > 0, 
                                           data['volume_support_up'], 
                                           -data['volume_support_down'])
    data['divergence_warning'] = data['price_up_volume_down'] - data['price_down_volume_up']
    data['volume_support_score'] = np.tanh(data['confirmation_strength']) - 0.3 * data['divergence_warning']
    
    # Regime classification
    data['volatility_regime'] = np.where(data['compression_intensity'] > 0.5, 'compression',
                                       np.where(data['expansion_magnitude'] > 0.3, 'expansion', 'transition'))
    
    # Dynamic regime weighting
    compression_mask = data['volatility_regime'] == 'compression'
    expansion_mask = data['volatility_regime'] == 'expansion'
    transition_mask = data['volatility_regime'] == 'transition'
    
    # Compression regime: focus on breakout signals and early momentum
    data['compression_signal'] = (data['strong_momentum_init'] * 0.6 + 
                                data['momentum_quality'] * 0.4)
    
    # Expansion regime: focus on momentum persistence and volume confirmation
    data['expansion_signal'] = (data['trend_persistence'] * 0.4 + 
                              data['momentum_quality'] * 0.3 + 
                              data['volume_support_score'] * 0.3)
    
    # Transition regime: adaptive sensitivity
    data['transition_signal'] = (data['compression_to_expansion'] * 0.5 - 
                               data['expansion_to_compression'] * 0.5 + 
                               data['momentum_accel_trend'] * 0.3)
    
    # Composite alpha signal with regime-adaptive weighting
    data['regime_adaptive_alpha'] = (
        compression_mask * data['compression_signal'] +
        expansion_mask * data['expansion_signal'] +
        transition_mask * data['transition_signal']
    )
    
    # Final quality filtering and normalization
    alpha = data['regime_adaptive_alpha'] * data['momentum_quality']
    alpha = alpha.fillna(0)
    
    # Remove any potential lookahead bias
    alpha = alpha.shift(1)  # Use yesterday's signal for today's prediction
    
    return alpha
