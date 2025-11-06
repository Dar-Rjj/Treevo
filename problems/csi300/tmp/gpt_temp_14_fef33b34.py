import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price differences and gaps
    data['gap'] = np.abs(data['open'] - data['close'].shift(1))
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['range'] = data['high'] - data['low']
    
    # Multi-Scale Gap Momentum Framework
    # Cross-Gap Momentum Divergence (simplified without sector/market data)
    data['gap_momentum_short'] = (data['close'] / data['close'].shift(5) - 1)
    data['gap_momentum_divergence'] = data['gap_momentum_short'] * data['gap'] / (data['range'] + 1e-8)
    
    # Volatility-Adjusted Gap Momentum
    data['gap_vol_ratio'] = data['range'] / (data['gap'] + 1e-8)
    data['gap_momentum_vol_adj'] = data['gap_momentum_divergence'] / (data['gap_vol_ratio'] + 1e-8)
    data['gap_momentum_efficiency'] = data['gap_momentum_vol_adj'] * np.sign(data['price_change'])
    
    # Volume-Confirmed Gap Momentum
    data['volume_ratio_5d'] = data['volume'] / (data['volume'].shift(5) + 1e-8) - 1
    data['gap_momentum_volume_align'] = data['gap_momentum_divergence'] * data['volume_ratio_5d']
    data['gap_volume_breakout'] = data['gap_momentum_volume_align'] * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Gap momentum persistence
    gap_sign = np.sign(data['gap_momentum_divergence'])
    data['gap_persistence'] = 0
    for i in range(4, len(data)):
        if i >= 4:
            current_sign = gap_sign.iloc[i]
            count = sum(gap_sign.iloc[i-4:i+1] == current_sign)
            data.iloc[i, data.columns.get_loc('gap_persistence')] = count
    
    data['gap_momentum_persistence'] = data['gap_volume_breakout'] * data['gap_persistence']
    
    # Multi-Timeframe Gap Dynamics
    # Short-Long Gap Fracture
    data['ultra_short_gap'] = np.abs(data['close'] - data['close'].shift(2))
    data['short_volatility'] = np.abs(data['close'] - data['close'].shift(1)) + np.abs(data['close'].shift(1) - data['close'].shift(2))
    data['ultra_short_momentum'] = data['ultra_short_gap'] / (data['short_volatility'] + 1e-8) * data['range'] / (data['gap'] + 1e-8)
    
    # Medium-term gap momentum (7-day)
    data['medium_gap'] = np.abs(data['close'] - data['close'].shift(7))
    data['medium_volatility'] = 0
    for i in range(6, len(data)):
        if i >= 6:
            vol_sum = sum(np.abs(data['close'].iloc[i-j] - data['close'].iloc[i-j-1]) for j in range(7))
            data.iloc[i, data.columns.get_loc('medium_volatility')] = vol_sum
    
    # Calculate rolling high/low for medium term
    data['medium_high'] = data['high'].rolling(window=7, min_periods=1).max()
    data['medium_low'] = data['low'].rolling(window=7, min_periods=1).min()
    data['medium_range'] = data['medium_high'] - data['medium_low']
    
    data['medium_momentum'] = data['medium_gap'] / (data['medium_volatility'] + 1e-8) * data['medium_range'] / (data['gap'] + 1e-8)
    
    data['gap_momentum_divergence_tf'] = (data['ultra_short_momentum'] - data['medium_momentum']) * np.sign(data['price_change'])
    
    # Cross-Gap Acceleration Framework
    data['cross_gap_acceleration'] = data['gap_momentum_divergence'] - data['gap_momentum_divergence'].shift(1)
    data['gap_volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    data['intraday_range_ratio'] = (data['close'] - data['open']) / (data['range'] + 1e-8)
    data['gap_acceleration_alignment'] = data['cross_gap_acceleration'] * data['gap_volume_acceleration'] * data['intraday_range_ratio']
    
    # Gap Microstructure Pressure Integration
    # Gap Opening Session Dynamics
    data['cross_gap_opening_pressure'] = ((data['open'] - data['low']) / (data['high'] - data['open'] + 1e-8) - 1) * np.sign(data['gap_momentum_divergence'])
    data['gap_opening_absorption'] = (np.abs(data['close'] - data['open']) / (data['gap'] + 1e-8)) * np.sign(data['gap_momentum_divergence'])
    data['gap_boundary_asymmetry'] = ((data['high'] - data['open']) - (data['open'] - data['low'])) / (data['range'] + 1e-8)
    data['quantum_gap_pressure'] = data['cross_gap_opening_pressure'] * data['gap_opening_absorption'] * data['gap_boundary_asymmetry']
    
    # Gap Volume Microstructure
    data['gap_volume_concentration'] = (data['amount'] / (data['amount'].shift(2) + data['amount'].shift(1) + data['amount'] + 1e-8)) * np.sign(data['gap_momentum_divergence'])
    data['gap_trade_size_momentum'] = ((data['amount'] / data['volume']) / (data['amount'].shift(1) / data['volume'].shift(1) + 1e-8) - 1) * np.sign(data['gap_momentum_divergence'])
    data['gap_microstructure_flow'] = data['gap_volume_concentration'] * data['gap_trade_size_momentum'] * (data['price_change'] / (data['range'] + 1e-8))
    
    # Gap Closing Session Asymmetry
    data['cross_gap_closing_pressure'] = (((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['range'] + 1e-8)) * np.sign(data['gap_momentum_divergence'])
    data['gap_session_completion'] = (np.abs(data['close'] - data['open']) / (data['range'] + 1e-8)) * np.sign(data['gap_momentum_divergence'])
    data['gap_closing_efficiency'] = data['cross_gap_closing_pressure'] * data['gap_session_completion'] * (data['volume'] / data['volume'].shift(1))
    
    # Dynamic Gap Regime Switching
    # Gap Volume Regime Classification
    data['volume_ratio_5d_regime'] = data['volume'] / data['volume'].shift(5)
    data['high_gap_volume'] = (data['volume_ratio_5d_regime'] > 1.5) & (data['gap_volume_acceleration'] > 0.1)
    data['low_gap_volume'] = (data['volume_ratio_5d_regime'] < 0.7) | (data['gap_volume_acceleration'] < -0.1)
    data['normal_gap_volume'] = ~(data['high_gap_volume'] | data['low_gap_volume'])
    
    # Cross-Gap Momentum Regime Detection
    data['strong_cross_gap'] = np.abs(data['gap_momentum_divergence']) > 0.8 * data['range']
    data['weak_cross_gap'] = np.abs(data['gap_momentum_divergence']) < 0.3 * data['range']
    data['transition_cross_gap'] = ~(data['strong_cross_gap'] | data['weak_cross_gap'])
    
    # Gap Regime Interaction Multipliers
    data['regime_multiplier'] = 1.0
    high_strong_mask = data['high_gap_volume'] & data['strong_cross_gap']
    high_weak_mask = data['high_gap_volume'] & data['weak_cross_gap']
    low_strong_mask = data['low_gap_volume'] & data['strong_cross_gap']
    low_weak_mask = data['low_gap_volume'] & data['weak_cross_gap']
    
    data.loc[high_strong_mask, 'regime_multiplier'] = 2.2
    data.loc[high_weak_mask, 'regime_multiplier'] = 1.4
    data.loc[low_strong_mask, 'regime_multiplier'] = 1.8
    data.loc[low_weak_mask, 'regime_multiplier'] = 0.6
    
    # Integrated Gap Signal Synthesis
    # Core Cross-Gap Momentum Signal
    data['base_gap_momentum'] = data['gap_momentum_divergence_tf'] * data['gap_momentum_persistence']
    data['gap_acceleration_enhanced'] = data['base_gap_momentum'] * data['gap_acceleration_alignment']
    data['regime_weighted_gap_momentum'] = data['gap_acceleration_enhanced'] * data['regime_multiplier']
    
    # Gap Microstructure Integration
    data['gap_pressure_aligned'] = data['regime_weighted_gap_momentum'] * data['quantum_gap_pressure']
    data['gap_flow_confirmed'] = data['gap_pressure_aligned'] * data['gap_microstructure_flow']
    data['gap_session_completed'] = data['gap_flow_confirmed'] * data['gap_closing_efficiency']
    
    # Gap Volume Dynamics Refinement
    data['volume_regime_weight'] = 1.0
    data.loc[data['high_gap_volume'], 'volume_regime_weight'] = 1.2
    data.loc[data['low_gap_volume'], 'volume_regime_weight'] = 0.8
    
    data['gap_volume_scaled'] = data['gap_session_completed'] * data['volume_regime_weight']
    data['gap_trade_size_aligned'] = data['gap_volume_scaled'] * data['gap_trade_size_momentum']
    data['gap_concentration_persistent'] = data['gap_trade_size_aligned'] * data['gap_volume_concentration']
    
    # Final Composite Gap Alpha Output
    data['final_gap_alpha'] = (data['gap_concentration_persistent'] * 
                              data['gap_persistence'] * 
                              data['gap_volume_acceleration'] * 
                              data['quantum_gap_pressure'] * 
                              data['regime_multiplier'] * 
                              data['gap_volume_concentration'])
    
    # Clean up and return
    result = data['final_gap_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
