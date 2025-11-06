import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Regime State Identification
    # Volatility Regime Classification
    data['ret'] = data['close'] / data['close'].shift(1) - 1
    data['vol_20'] = data['ret'].rolling(window=20, min_periods=1).std()
    data['vol_60'] = data['ret'].rolling(window=60, min_periods=1).std()
    data['vol_regime'] = np.where(data['vol_20'] > data['vol_60'], 'high', 'normal')
    
    # Intraday Volatility Dynamics
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    data['vol_persistence'] = data['intraday_vol'] - (data['high'].shift(5) - data['low'].shift(5)) / data['close'].shift(5)
    
    # Volume Regime Assessment
    data['vol_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_surge'] = data['volume'] > (1.5 * data['vol_ma_5'])
    data['volume_drought'] = data['volume'] < (0.7 * data['vol_ma_5'])
    
    # Microstructure Velocity Components
    # Order Flow Asymmetry Dynamics
    data['order_flow_imbalance'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    data['order_flow_imbalance_ma_5'] = data['order_flow_imbalance'].rolling(window=5, min_periods=1).mean()
    data['order_flow_momentum'] = data['order_flow_imbalance'] - data['order_flow_imbalance_ma_5']
    
    data['max_oc'] = np.maximum(data['open'], data['close'])
    data['min_oc'] = np.minimum(data['open'], data['close'])
    data['order_flow_rejection'] = data['order_flow_momentum'] * ((data['high'] - data['max_oc']) - (data['min_oc'] - data['low']))
    
    # Volatility-Adjusted Momentum
    data['clean_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['vol_adj_momentum'] = data['clean_momentum'] / data['intraday_vol']
    
    vol_diff_3 = data['intraday_vol'] - (data['high'].shift(3) - data['low'].shift(3)) / data['close'].shift(3)
    data['regime_transition_momentum'] = (data['close'] / data['close'].shift(3) - 1) * np.sign(vol_diff_3)
    
    # Volume-Velocity Integration
    data['liquidity_absorption'] = data['volume'] / (data['high'] - data['low'])
    data['liquidity_absorption_ma_5'] = data['liquidity_absorption'].rolling(window=5, min_periods=1).mean()
    data['absorption_momentum'] = data['liquidity_absorption'] / data['liquidity_absorption_ma_5']
    data['volume_vol_efficiency'] = data['volume'] / (data['high'] - data['low'])
    
    # Asymmetric Efficiency Framework
    # Multi-timeframe Efficiency Patterns
    data['short_term_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['medium_term_efficiency'] = (data['close'] - data['close'].shift(5)) / (data['high'].rolling(window=5, min_periods=1).max() - data['low'].rolling(window=5, min_periods=1).min())
    data['efficiency_divergence'] = data['short_term_efficiency'] - data['medium_term_efficiency']
    
    # Rejection Bias Analysis
    data['net_asymmetric_rejection'] = (data['high'] - data['max_oc']) - (data['min_oc'] - data['low'])
    data['rejection_bias_ratio'] = (data['high'] - data['max_oc']) / (data['min_oc'] - data['low'])
    data['vol_compression_signal'] = ((data['high'] - data['close']) / (data['close'] - data['low'])) * data['intraday_vol']
    
    # Temporal Validation Patterns
    def calculate_persistence(series, window):
        signs = np.sign(series.diff())
        persistence = signs.rolling(window=window, min_periods=1).apply(lambda x: (x == x.shift(1)).sum() / (window-1) if len(x) == window else np.nan, raw=False)
        return persistence
    
    data['efficiency_persistence'] = calculate_persistence(data['short_term_efficiency'], 3)
    data['momentum_consistency'] = calculate_persistence(data['clean_momentum'], 3)
    
    # Cross-Regime Velocity Enhancement
    # Regime-Specific Multipliers
    data['rejection_multiplier'] = np.where(data['vol_regime'] == 'high', 1.3, 1.0)
    data['efficiency_multiplier'] = np.where(data['vol_regime'] == 'normal', 1.0, 0.8)
    data['volume_surge_multiplier'] = np.where(data['volume_surge'], 1.2, 1.0)
    data['volume_drought_multiplier'] = np.where(data['volume_drought'], 0.8, 1.0)
    
    # Divergence-Confirmed Signals
    data['order_flow_alignment'] = np.sign(data['net_asymmetric_rejection']) * np.sign(data['order_flow_momentum'])
    
    eff_diff = data['short_term_efficiency'] - data['short_term_efficiency'].shift(1)
    vol_diff = data['volume'] / data['volume'].shift(1) - 1
    data['volume_efficiency_divergence'] = np.sign(eff_diff) * np.sign(vol_diff)
    
    vol_vol_diff = data['intraday_vol'] - data['intraday_vol'].shift(1)
    data['volatility_volume_divergence'] = vol_diff * vol_vol_diff
    
    # Multi-timeframe Integration
    data['weekly_vol_momentum'] = (data['close'] / data['close'].shift(5) - 1) / data['intraday_vol']
    
    monthly_vol_diff = data['intraday_vol'] - (data['high'].shift(21) - data['low'].shift(21)) / data['close'].shift(21)
    data['monthly_vol_regime_shift'] = np.sign(monthly_vol_diff) * (data['close'] / data['close'].shift(21) - 1)
    
    data['range_dynamics'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Adaptive Alpha Construction
    # Core Velocity Factors
    data['microstructure_velocity'] = data['order_flow_rejection'] * data['vol_adj_momentum']
    data['volume_absorption_velocity'] = data['clean_momentum'] * data['absorption_momentum']
    data['efficiency_breakout'] = data['vol_compression_signal'] * data['short_term_efficiency']
    data['regime_transition_velocity'] = data['regime_transition_momentum'] * data['vol_persistence']
    
    # Divergence-Enhanced Components
    data['aligned_microstructure'] = data['microstructure_velocity'] * data['order_flow_alignment'] * data['rejection_multiplier']
    data['volume_efficiency_momentum'] = data['volume_absorption_velocity'] * data['volume_efficiency_divergence'] * data['volume_surge_multiplier']
    data['volatility_enhanced_breakout'] = data['efficiency_breakout'] * data['volatility_volume_divergence'] * data['efficiency_multiplier']
    data['multi_timeframe_regime'] = data['regime_transition_velocity'] * data['monthly_vol_regime_shift'] * data['volume_drought_multiplier']
    
    # Final Alpha Synthesis
    primary_factor = data['aligned_microstructure'] * data['momentum_consistency']
    secondary_factor = data['volume_efficiency_momentum'] * data['efficiency_persistence']
    tertiary_factor = data['volatility_enhanced_breakout'] * data['weekly_vol_momentum']
    quaternary_factor = data['multi_timeframe_regime'] * data['range_dynamics']
    
    # Combine factors with weights
    alpha = (0.4 * primary_factor + 
             0.3 * secondary_factor + 
             0.2 * tertiary_factor + 
             0.1 * quaternary_factor)
    
    return alpha
