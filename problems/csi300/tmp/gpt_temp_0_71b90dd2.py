import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Directional Volume Pressure Analysis
    # Intraday Price Range Efficiency
    data['intraday_range'] = data['high'] - data['low']
    data['price_change'] = data['close'] - data['open']
    
    # Calculate directional efficiency
    data['upside_efficiency'] = np.where(
        data['price_change'] > 0,
        data['price_change'] / np.maximum(data['intraday_range'], 1e-8),
        0
    )
    data['downside_efficiency'] = np.where(
        data['price_change'] < 0,
        -data['price_change'] / np.maximum(data['intraday_range'], 1e-8),
        0
    )
    data['abs_efficiency'] = np.abs(data['price_change']) / np.maximum(data['intraday_range'], 1e-8)
    
    # Volume-Weighted Price Pressure
    def calculate_pressure_intensity(window_data, direction='upside'):
        if direction == 'upside':
            pressure = (window_data['high'] - window_data['close']) * window_data['volume'] * window_data['abs_efficiency']
        else:
            pressure = (window_data['close'] - window_data['low']) * window_data['volume'] * window_data['abs_efficiency']
        return pressure.sum()
    
    # Calculate rolling pressure intensities
    upside_pressure = []
    downside_pressure = []
    
    for i in range(len(data)):
        if i < 4:
            upside_pressure.append(np.nan)
            downside_pressure.append(np.nan)
            continue
            
        window_data = data.iloc[i-4:i+1]
        upside_pressure.append(calculate_pressure_intensity(window_data, 'upside'))
        downside_pressure.append(calculate_pressure_intensity(window_data, 'downside'))
    
    data['upside_pressure'] = upside_pressure
    data['downside_pressure'] = downside_pressure
    data['net_pressure_divergence'] = (data['upside_pressure'] - data['downside_pressure']) / \
                                     np.maximum(data['upside_pressure'] + data['downside_pressure'], 1e-8)
    
    # Multi-Timeframe Pressure Consistency
    data['short_term_pressure'] = data['net_pressure_divergence'].rolling(window=3, min_periods=3).mean()
    data['medium_term_pressure'] = data['net_pressure_divergence'].rolling(window=8, min_periods=8).mean()
    
    data['pressure_consistency'] = np.sign(data['short_term_pressure']) * \
                                  np.sign(data['medium_term_pressure']) * \
                                  np.minimum(np.abs(data['short_term_pressure']), 
                                           np.abs(data['medium_term_pressure']))
    
    # 2. Price-Momentum Regime Detection
    # Acceleration-Deceleration Analysis
    data['fast_momentum'] = data['close'] - data['close'].shift(2)
    data['slow_momentum'] = data['close'] - data['close'].shift(6)
    data['momentum_regime'] = np.sign(data['fast_momentum']) * \
                             np.sign(data['slow_momentum']) * \
                             (np.abs(data['fast_momentum']) - np.abs(data['slow_momentum']))
    
    # Volatility-Adjusted Momentum Strength
    data['recent_volatility'] = (data['high'] - data['low']).rolling(window=5, min_periods=5).mean()
    data['momentum_vol_ratio'] = (data['close'] - data['close'].shift(5)) / np.maximum(data['recent_volatility'], 1e-8)
    data['regime_confirmation'] = data['momentum_regime'] * data['momentum_vol_ratio']
    
    # Breakout Momentum Enhancement
    data['price_range_10d'] = data['high'].rolling(window=10, min_periods=10).max() - \
                             data['low'].rolling(window=10, min_periods=10).min()
    data['current_position'] = (data['close'] - data['low'].rolling(window=10, min_periods=10).min()) / \
                              np.maximum(data['price_range_10d'], 1e-8)
    data['range_breakout_multiplier'] = 1 + np.abs(data['current_position'] - 0.5) * 2
    
    # 3. Volume Efficiency Transition Analysis
    # Efficiency Persistence Tracking
    data['efficiency_sign'] = np.sign(data['price_change'])
    data['consecutive_days'] = 0
    
    for i in range(1, len(data)):
        if data['efficiency_sign'].iloc[i] == data['efficiency_sign'].iloc[i-1]:
            data.loc[data.index[i], 'consecutive_days'] = data['consecutive_days'].iloc[i-1] + 1
    
    data['efficiency_momentum'] = data['consecutive_days'] * data['abs_efficiency'].rolling(window=5, min_periods=5).mean()
    
    # Volume Concentration Shifts
    data['volume_clustering'] = data['volume'] / data['volume'].shift(1).rolling(window=5, min_periods=5).mean()
    data['efficiency_volume_corr'] = data['abs_efficiency'] * data['volume_clustering']
    
    # Detect regime shifts (when correlation changes sign)
    data['regime_shift'] = (np.sign(data['efficiency_volume_corr']) != np.sign(data['efficiency_volume_corr'].shift(1))).astype(int)
    
    # Transition Confirmation Signals
    data['pre_transition_pressure'] = data['pressure_consistency'].shift(2)
    data['post_transition_momentum'] = data['momentum_regime'].shift(-1)  # Note: This uses future data, but we'll handle this differently
    
    # Instead of using future data, use current momentum regime for transition strength
    data['transition_strength'] = np.abs(data['pre_transition_pressure']) * np.abs(data['momentum_regime'])
    
    # 4. Signal Integration with Dynamic Weighting
    # Core Divergence Signal
    data['base_signal'] = data['pressure_consistency'] * data['regime_confirmation']
    data['volume_adjusted_signal'] = data['base_signal'] * data['efficiency_momentum']
    data['range_scaled_signal'] = data['volume_adjusted_signal'] * data['range_breakout_multiplier']
    
    # Regime Transition Enhancement
    data['transition_multiplier'] = np.where(
        data['regime_shift'] == 1,
        1 + data['transition_strength'],
        1
    )
    data['transition_enhanced_signal'] = data['range_scaled_signal'] * data['transition_multiplier']
    data['persistence_bonus'] = data['efficiency_momentum'] * 0.1
    data['enhanced_signal'] = data['transition_enhanced_signal'] + data['persistence_bonus']
    
    # Dynamic Signal Refinement
    data['vol_normalized_signal'] = data['enhanced_signal'] / np.maximum(data['recent_volatility'], 1e-8)
    data['volume_scaled_signal'] = data['vol_normalized_signal'] * data['volume_clustering']
    data['direction_confirmed_signal'] = data['volume_scaled_signal'] * np.sign(data['net_pressure_divergence'])
    
    # 5. Multi-Timeframe Signal Convergence
    # Short-term component (3-day)
    data['short_term_pressure_3d'] = data['net_pressure_divergence'].rolling(window=3, min_periods=3).mean()
    data['short_term_momentum_3d'] = data['momentum_regime'].rolling(window=3, min_periods=3).mean()
    data['short_term_signal'] = data['short_term_pressure_3d'] * data['short_term_momentum_3d']
    
    # Medium-term component (8-day)
    data['medium_term_pressure_8d'] = data['net_pressure_divergence'].rolling(window=8, min_periods=8).mean()
    data['medium_term_momentum_8d'] = data['momentum_regime'].rolling(window=8, min_periods=8).mean()
    data['medium_term_signal'] = data['medium_term_pressure_8d'] * data['medium_term_momentum_8d']
    
    # Multi-timeframe integration
    data['timeframe_alignment'] = np.sign(data['short_term_signal']) * np.sign(data['medium_term_signal'])
    data['combined_signal_strength'] = (data['short_term_signal'] + data['medium_term_signal']) * data['timeframe_alignment']
    data['final_convergence'] = data['combined_signal_strength'] * data['timeframe_alignment']
    
    # 6. Final Alpha Generation
    # Signal Transformation
    data['alpha_factor'] = np.tanh(data['final_convergence'])
    
    # Clean up and return the alpha factor series
    alpha_series = data['alpha_factor'].copy()
    alpha_series.name = 'price_volume_divergence_momentum'
    
    return alpha_series
