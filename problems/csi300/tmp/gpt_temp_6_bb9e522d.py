import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price changes and ratios
    data['close_ret_2d'] = data['close'] / data['close'].shift(2) - 1
    data['high_low_range'] = data['high'] - data['low']
    data['close_open_range'] = abs(data['close'] - data['open'])
    data['open_close_prev_range'] = abs(data['open'] - data['close'].shift(1))
    data['close_prev_range'] = abs(data['close'] - data['close'].shift(1))
    
    # Market and Sector proxies (using rolling averages as proxies)
    data['market_close'] = data['close'].rolling(window=20).mean()
    data['sector_close'] = data['close'].rolling(window=10).mean()
    
    # Multi-Asset Momentum Fracture
    data['market_ret_2d'] = data['market_close'] / data['market_close'].shift(2) - 1
    data['sector_ret_2d'] = data['sector_close'] / data['sector_close'].shift(2) - 1
    
    # Avoid division by zero
    market_denom = np.where(abs(data['market_ret_2d']) > 1e-8, data['market_ret_2d'], 1e-8)
    sector_denom = np.where(abs(data['sector_ret_2d']) > 1e-8, data['sector_ret_2d'], 1e-8)
    
    data['stock_market_fracture'] = (data['close_ret_2d'] / market_denom) * (data['high_low_range'] / np.where(data['close_open_range'] > 0, data['close_open_range'], 1))
    data['stock_sector_fracture'] = (data['close_ret_2d'] / sector_denom) * (data['high_low_range'] / np.where(data['open_close_prev_range'] > 0, data['open_close_prev_range'], 1))
    data['fracture_momentum_divergence'] = data['stock_market_fracture'] - data['stock_sector_fracture']
    
    # Volatility-Adapted Fracture Momentum
    vol_scale = data['high_low_range'] / np.where(data['close_prev_range'] > 0, data['close_prev_range'], 1)
    data['fracture_momentum_vol_scaled'] = data['fracture_momentum_divergence'] * vol_scale
    
    volume_ratio = data['volume'] / np.where(data['volume'].shift(2) > 0, data['volume'].shift(2), 1)
    data['volume_anchored_fracture'] = data['fracture_momentum_vol_scaled'] * volume_ratio
    
    amount_ratio = data['amount'] / np.where(data['amount'].shift(2) > 0, data['amount'].shift(2), 1)
    data['amount_anchored_fracture'] = data['volume_anchored_fracture'] * amount_ratio
    
    # Fracture Momentum Persistence
    fracture_sign = np.sign(data['fracture_momentum_divergence'])
    data['fracture_directional_persistence'] = fracture_sign.rolling(window=3).apply(lambda x: np.sum(x == x.iloc[-1]) if len(x) == 3 else np.nan, raw=False)
    
    fracture_strength = abs(data['fracture_momentum_divergence']) / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)
    data['fracture_strength_persistence'] = data['fracture_directional_persistence'] * fracture_strength
    
    volume_ratio_1d = data['volume'] / np.where(data['volume'].shift(1) > 0, data['volume'].shift(1), 1)
    data['volume_anchored_momentum_persistence'] = data['fracture_strength_persistence'] * volume_ratio_1d
    
    # Multi-Timeframe Fracture Dynamics
    data['ultra_short_fracture'] = ((data['close'] - data['close'].shift(1)) / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)) * (data['close_open_range'] / np.where(data['high_low_range'] > 0, data['high_low_range'], 1))
    
    # Long-term fracture (6-day)
    data['high_6d'] = data['high'].rolling(window=6).max()
    data['low_6d'] = data['low'].rolling(window=6).min()
    data['long_term_fracture'] = ((data['close'] - data['close'].shift(6)) / np.where((data['high_6d'] - data['low_6d']) > 0, (data['high_6d'] - data['low_6d']), 1)) * (data['open_close_prev_range'] / np.where(data['high_low_range'] > 0, data['high_low_range'], 1))
    
    close_dir = np.sign(data['close'] - data['close'].shift(1))
    data['fracture_momentum_gap'] = data['ultra_short_fracture'] - data['long_term_fracture'] * close_dir
    
    # Fracture Momentum Acceleration
    data['fracture_momentum_velocity'] = data['fracture_momentum_divergence'] - data['fracture_momentum_divergence'].shift(1)
    
    volume_momentum_1 = data['volume'] / np.where(data['volume'].shift(1) > 0, data['volume'].shift(1), 1)
    volume_momentum_2 = data['volume'].shift(1) / np.where(data['volume'].shift(2) > 0, data['volume'].shift(2), 1)
    data['volume_momentum_velocity'] = volume_momentum_1 - volume_momentum_2
    
    close_open_ratio = (data['close'] - data['open']) / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)
    data['fracture_momentum_alignment'] = data['fracture_momentum_velocity'] * data['volume_momentum_velocity'] * close_open_ratio
    
    # Fracture Pressure Dynamics
    lower_pressure = ((data['close'] - data['low']) / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)) * (data['close_open_range'] / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)) * np.sign(data['fracture_momentum_divergence'])
    upper_pressure = ((data['high'] - data['close']) / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)) * (data['open_close_prev_range'] / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)) * np.sign(data['fracture_momentum_divergence'])
    
    amount_volume_ratio = data['amount'] / np.where(data['volume'] > 0, data['volume'], 1)
    data['fracture_pressure_differential'] = (lower_pressure - upper_pressure) * amount_volume_ratio
    
    # Microstructure Anchoring Framework
    # Opening Session Anchoring
    opening_anchor_pressure = ((data['open'] - data['low']) / np.where((data['high'] - data['open']) > 0, (data['high'] - data['open']), 1) - 1) * np.sign(data['fracture_momentum_divergence']) * (data['open_close_prev_range'] / np.where(data['high_low_range'] > 0, data['high_low_range'], 1))
    opening_anchor_absorption = (data['close_open_range'] / np.where(data['open_close_prev_range'] > 0, data['open_close_prev_range'], 1)) * np.sign(data['fracture_momentum_divergence'])
    data['opening_anchor_alignment'] = opening_anchor_pressure * opening_anchor_absorption
    
    # Intraday Volume Anchoring
    volume_anchor_concentration = (data['amount'] / np.where((data['amount'].shift(2) + data['amount']) > 0, (data['amount'].shift(2) + data['amount']), 1)) * np.sign(data['fracture_momentum_divergence']) * (data['close_open_range'] / np.where(data['high_low_range'] > 0, data['high_low_range'], 1))
    
    trade_size_current = data['amount'] / np.where(data['volume'] > 0, data['volume'], 1)
    trade_size_prev = data['amount'].shift(1) / np.where(data['volume'].shift(1) > 0, data['volume'].shift(1), 1)
    trade_size_anchor_dynamics = (trade_size_current / np.where(trade_size_prev > 0, trade_size_prev, 1) - 1) * np.sign(data['fracture_momentum_divergence'])
    
    close_prev_ratio = (data['close'] - data['close'].shift(1)) / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)
    data['microstructure_anchor_flow'] = volume_anchor_concentration * trade_size_anchor_dynamics * close_prev_ratio
    
    # Closing Session Anchoring
    closing_anchor_pressure = (((data['close'] - data['low']) - (data['high'] - data['close'])) / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)) * np.sign(data['fracture_momentum_divergence']) * (data['close_open_range'] / np.where(data['high_low_range'] > 0, data['high_low_range'], 1))
    closing_anchor_completion = (data['close_open_range'] / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)) * np.sign(data['fracture_momentum_divergence'])
    data['closing_anchor_efficiency'] = closing_anchor_pressure * closing_anchor_completion * volume_ratio
    
    # Dynamic Anchoring Regime Classification
    volume_ratio_3d = data['volume'] / np.where(data['volume'].shift(3) > 0, data['volume'].shift(3), 1)
    
    high_volume_anchoring = (volume_ratio_3d > 1.5) & (data['volume_momentum_velocity'] > 0.1)
    low_volume_anchoring = (volume_ratio_3d < 0.7) | (data['volume_momentum_velocity'] < -0.1)
    normal_volume_anchoring = ~high_volume_anchoring & ~low_volume_anchoring
    
    fracture_strength_threshold = abs(data['fracture_momentum_divergence']) / np.where(data['high_low_range'] > 0, data['high_low_range'], 1)
    strong_fracture = fracture_strength_threshold > 0.8
    weak_fracture = fracture_strength_threshold < 0.3
    transition_fracture = ~strong_fracture & ~weak_fracture
    
    # Anchoring Regime Multipliers
    regime_multiplier = np.ones(len(data))
    regime_multiplier[high_volume_anchoring & strong_fracture] = 2.2
    regime_multiplier[high_volume_anchoring & weak_fracture] = 1.4
    regime_multiplier[low_volume_anchoring & strong_fracture] = 1.8
    regime_multiplier[low_volume_anchoring & weak_fracture] = 0.6
    regime_multiplier[normal_volume_anchoring & transition_fracture] = 1.0
    
    # Integrated Anchoring Signal Synthesis
    # Core Anchoring Signal
    base_anchoring = data['fracture_momentum_gap'] * data['volume_anchored_momentum_persistence']
    anchoring_acceleration_enhanced = base_anchoring * data['fracture_momentum_alignment']
    regime_weighted_anchoring = anchoring_acceleration_enhanced * regime_multiplier
    
    # Microstructure Anchoring Integration
    anchoring_pressure_aligned = regime_weighted_anchoring * data['opening_anchor_alignment']
    anchoring_flow_confirmed = anchoring_pressure_aligned * data['microstructure_anchor_flow']
    anchoring_session_completed = anchoring_flow_confirmed * data['closing_anchor_efficiency']
    
    # Volume Anchoring Dynamics Refinement
    volume_regime_weight = np.where(high_volume_anchoring, 1.2, np.where(low_volume_anchoring, 0.8, 1.0))
    anchoring_volume_scaled = anchoring_session_completed * volume_regime_weight
    anchoring_trade_size_aligned = anchoring_volume_scaled * trade_size_anchor_dynamics
    anchoring_concentration_persistent = anchoring_trade_size_aligned * volume_anchor_concentration
    
    # Final Composite Anchoring Alpha
    # Apply fracture momentum persistence for signal stability
    final_signal = anchoring_concentration_persistent * data['fracture_strength_persistence']
    
    # Scale by fracture momentum velocity for dynamic adjustment
    final_signal = final_signal * (1 + data['fracture_momentum_velocity'])
    
    # Incorporate opening anchor pressure for microstructure context
    final_signal = final_signal * (1 + opening_anchor_pressure)
    
    # Integrate anchoring regime switching multipliers for adaptive behavior
    final_signal = final_signal * regime_multiplier
    
    # Apply volume anchor concentration for flow dynamics
    final_signal = final_signal * (1 + volume_anchor_concentration)
    
    # Clean up any infinite or NaN values
    final_signal = final_signal.replace([np.inf, -np.inf], np.nan)
    
    return final_signal
