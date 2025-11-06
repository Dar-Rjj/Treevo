import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Fractal Regime Classification
    # Volatility Fractal Dynamics
    data['ret'] = data['close'] / data['close'].shift(1) - 1
    data['vol_20'] = data['ret'].rolling(window=20, min_periods=20).std()
    data['vol_60'] = data['ret'].rolling(window=60, min_periods=60).std()
    data['vol_regime'] = np.where(data['vol_20'] > data['vol_60'], 1.3, 1.0)
    
    # Volume Fractal Assessment
    data['vol_ma_5'] = (data['volume'].shift(4) + data['volume'].shift(3) + 
                        data['volume'].shift(2) + data['volume'].shift(1) + data['volume']) / 5
    data['vol_surge'] = np.where(data['volume'] > 1.5 * data['vol_ma_5'], 1.2, 1.0)
    data['vol_drought'] = np.where(data['volume'] < 0.7 * data['vol_ma_5'], 0.8, 1.0)
    data['vol_transition'] = np.sign(data['volume'] - (data['volume'].shift(1) + 
                                                      data['volume'].shift(2) + data['volume'].shift(3)) / 3)
    
    # Range Fractal Patterns
    data['range'] = data['high'] - data['low']
    data['range_ratio'] = data['range'] / data['range'].shift(1)
    data['range_expansion'] = np.where(data['range_ratio'] > 1.2, 1, 0)
    data['range_contraction'] = np.where(data['range_ratio'] < 0.8, 1, 0)
    data['vol_transition_signal'] = np.sign(data['range'] - (data['range'].shift(1) + 
                                                            data['range'].shift(2) + data['range'].shift(3)) / 3)
    
    # Asymmetric Efficiency Decay Framework
    # Multi-timeframe Rejection Decay
    data['upside_rejection'] = (data['high'] - np.maximum(data['open'], data['close'])) * data['volume'] / data['volume'].shift(3)
    data['downside_rejection'] = (np.minimum(data['open'], data['close']) - data['low']) * data['volume'] / data['volume'].shift(3)
    data['net_rejection'] = data['upside_rejection'] - data['downside_rejection']
    data['rejection_momentum'] = data['net_rejection'] / data['net_rejection'].shift(3) - 1
    
    # Fractal Efficiency Dynamics
    data['short_term_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['eff_decay_transition'] = np.abs(data['close'] - data['open']) / data['range'] / (
        np.abs(data['close'].shift(1) - data['open'].shift(1)) / data['range'].shift(1)) - 1
    data['eff_momentum'] = data['short_term_eff'] / data['short_term_eff'].shift(1) - 1
    data['range_capture_eff'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / data['range']
    
    # Temporal Decay Patterns
    data['high_capture_decay'] = (data['high'] - data['close']) / data['range'] * data['volume'] / data['volume'].shift(2)
    data['low_capture_decay'] = (data['close'] - data['low']) / data['range'] * data['volume'] / data['volume'].shift(2)
    
    # Volume-Absorption Velocity Decay
    # Liquidity Absorption Dynamics
    data['trade_eff'] = data['amount'] / data['range']
    data['liquidity_absorption'] = data['volume'] / data['range']
    data['abs_momentum'] = data['liquidity_absorption'] / data['liquidity_absorption'].rolling(window=5, min_periods=5).mean()
    data['eff_adj_absorption'] = data['trade_eff'] * data['abs_momentum']
    
    # Asymmetric Volume Velocity Decay
    data['upside_vol_intensity'] = data['volume'] / (data['high'] - data['open'])
    data['downside_vol_intensity'] = data['volume'] / (data['open'] - data['low'])
    data['vol_asymmetry_ratio'] = data['upside_vol_intensity'] / data['downside_vol_intensity']
    data['gap_vol_absorption'] = np.abs(data['close'] - data['open']) * data['volume'] / data['range']
    
    # Volume Acceleration Decay
    data['vol_acceleration'] = (data['volume'] / data['volume'].shift(3)) ** (1/3) - 1
    data['vol_flow_consistency'] = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['volume'].shift(1) - data['volume'].shift(2))
    data['vol_range_coherence'] = data['volume'] * data['range'] / (data['volume'].shift(3) * data['range'].shift(3))
    data['vol_weighted_eff'] = data['short_term_eff'] * data['volume']
    
    # Fractal Momentum Construction
    # Multi-scale Momentum Patterns
    data['clean_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_accel'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3) - (
        data['close'].shift(3) - data['close'].shift(6)) / data['close'].shift(6)
    data['momentum_eff_ratio'] = data['clean_momentum'] / (data['range'] / data['close'].shift(1))
    data['consecutive_return_decay'] = data['close'] / data['close'].shift(1) - data['close'].shift(1) / data['close'].shift(2)
    
    # Breakout Velocity Decay
    data['upside_breakout'] = data['high'] / data['high'].shift(1) - 1
    data['downside_breakout'] = data['low'] / data['low'].shift(1) - 1
    data['breakout_asymmetry'] = data['upside_breakout'] - data['downside_breakout']
    data['vol_enhanced_breakout'] = (data['high'] - data['high'].shift(1)) * data['range']
    
    # Regime-Enhanced Momentum Decay
    data['order_flow_momentum'] = data['clean_momentum'] * data['vol_flow_consistency']
    data['eff_weighted_velocity'] = data['short_term_accel'] * data['short_term_eff']
    data['vol_confirmed_momentum'] = data['clean_momentum'] * data['vol_acceleration']
    data['asymmetric_breakout_momentum'] = data['breakout_asymmetry'] * data['vol_asymmetry_ratio']
    
    # Divergence and Decay Validation
    # Microstructure Divergence Decay
    data['eff_vol_divergence'] = np.sign(data['short_term_eff'] - data['short_term_eff'].shift(1)) * np.sign(data['vol_acceleration'])
    data['rejection_vol_alignment'] = np.sign(data['net_rejection']) * data['vol_transition']
    data['price_vol_divergence'] = np.sign(data['clean_momentum']) * np.sign(data['vol_acceleration'])
    data['absorption_eff_divergence'] = np.sign(data['eff_adj_absorption']) * np.sign(data['eff_momentum'])
    
    # Range Dynamics Decay
    data['vol_regime_shift'] = data['range'] / data['range'].shift(3)
    data['mean_reversion_strength'] = 1 - np.abs(data['close'] - data['close'].shift(1)) / data['range']
    
    # Multi-timeframe Decay Validation
    for i in range(2, len(data)):
        data.loc[data.index[i], 'eff_persistence'] = (
            (np.sign(data.loc[data.index[i], 'short_term_eff'] - data.loc[data.index[i-1], 'short_term_eff']) == 
             np.sign(data.loc[data.index[i-1], 'short_term_eff'] - data.loc[data.index[i-2], 'short_term_eff'])) +
            (np.sign(data.loc[data.index[i-1], 'short_term_eff'] - data.loc[data.index[i-2], 'short_term_eff']) == 
             np.sign(data.loc[data.index[i-2], 'short_term_eff'] - data.loc[data.index[i-3], 'short_term_eff'])) +
            (np.sign(data.loc[data.index[i-2], 'short_term_eff'] - data.loc[data.index[i-3], 'short_term_eff']) == 
             np.sign(data.loc[data.index[i-3], 'short_term_eff'] - data.loc[data.index[i-4], 'short_term_eff']))
        ) / 3
        
        data.loc[data.index[i], 'momentum_consistency'] = (
            (np.sign(data.loc[data.index[i], 'clean_momentum']) == np.sign(data.loc[data.index[i-1], 'clean_momentum'])) +
            (np.sign(data.loc[data.index[i-1], 'clean_momentum']) == np.sign(data.loc[data.index[i-2], 'clean_momentum'])) +
            (np.sign(data.loc[data.index[i-2], 'clean_momentum']) == np.sign(data.loc[data.index[i-3], 'clean_momentum']))
        ) / 3
        
        data.loc[data.index[i], 'vol_flow_persistence'] = (
            (np.sign(data.loc[data.index[i], 'volume'] - data.loc[data.index[i-1], 'volume']) == 
             np.sign(data.loc[data.index[i-1], 'volume'] - data.loc[data.index[i-2], 'volume'])) +
            (np.sign(data.loc[data.index[i-1], 'volume'] - data.loc[data.index[i-2], 'volume']) == 
             np.sign(data.loc[data.index[i-2], 'volume'] - data.loc[data.index[i-3], 'volume'])) +
            (np.sign(data.loc[data.index[i-2], 'volume'] - data.loc[data.index[i-3], 'volume']) == 
             np.sign(data.loc[data.index[i-3], 'volume'] - data.loc[data.index[i-4], 'volume']))
        ) / 3
        
        data.loc[data.index[i], 'rejection_persistence'] = (
            (np.sign(data.loc[data.index[i], 'net_rejection']) == np.sign(data.loc[data.index[i-1], 'net_rejection'])) +
            (np.sign(data.loc[data.index[i-1], 'net_rejection']) == np.sign(data.loc[data.index[i-2], 'net_rejection'])) +
            (np.sign(data.loc[data.index[i-2], 'net_rejection']) == np.sign(data.loc[data.index[i-3], 'net_rejection']))
        ) / 3
    
    # Fractal Decay Alpha Synthesis
    # Core Velocity Decay Components
    data['rejection_vol_velocity'] = data['net_rejection'] * data['vol_acceleration']
    data['vol_absorption_velocity'] = data['clean_momentum'] * data['eff_adj_absorption']
    data['eff_breakout_velocity'] = data['breakout_asymmetry'] * data['short_term_eff']
    data['asymmetric_momentum_velocity'] = data['short_term_accel'] * data['vol_asymmetry_ratio']
    
    # Fractal Regime Enhancement
    data['rejection_vol_velocity'] *= data['vol_regime']
    data['vol_absorption_velocity'] *= data['vol_surge'] * data['vol_drought']
    
    # Decay-Confirmed Signals
    data['micro_confirmed_velocity'] = data['rejection_vol_velocity'] * data['rejection_vol_alignment']
    data['vol_eff_velocity'] = data['vol_absorption_velocity'] * data['absorption_eff_divergence']
    data['breakout_eff_velocity'] = data['eff_breakout_velocity'] * data['eff_vol_divergence']
    data['range_enhanced_momentum'] = data['asymmetric_momentum_velocity'] * (data['range_expansion'] - data['range_contraction'])
    
    # Final Fractal Alpha Construction
    data['primary_factor'] = data['micro_confirmed_velocity'] * data['vol_range_coherence']
    data['secondary_factor'] = data['vol_eff_velocity'] * data['momentum_consistency']
    data['tertiary_factor'] = data['breakout_eff_velocity'] * data['eff_persistence']
    data['quaternary_factor'] = data['range_enhanced_momentum'] * data['mean_reversion_strength']
    
    # Composite Alpha
    alpha = (data['primary_factor'] + data['secondary_factor'] + 
             data['tertiary_factor'] + data['quaternary_factor']) / 4
    
    return alpha
