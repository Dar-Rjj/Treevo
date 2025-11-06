import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Acceleration Framework
    # Price Acceleration Components
    data['intraday_acc'] = ((data['high'] - data['close']) / data['close']) - ((data['low'] - data['close']) / data['close'])
    data['short_term_acc'] = ((data['high'] - data['close'].shift(3)) / data['close'].shift(3)) - ((data['low'] - data['close'].shift(3)) / data['close'].shift(3))
    data['medium_term_acc'] = ((data['high'] - data['close'].shift(8)) / data['close'].shift(8)) - ((data['low'] - data['close'].shift(8)) / data['close'].shift(8))
    
    # Acceleration Divergence Detection
    data['intraday_short_div'] = data['intraday_acc'] - data['short_term_acc']
    data['short_medium_div'] = data['short_term_acc'] - data['medium_term_acc']
    data['net_acc_div'] = data['intraday_short_div'] - data['short_medium_div']
    
    # Volatility-Adaptive Divergence Enhancement
    # Volatility Measurement
    data['daily_range_vol'] = (data['high'] - data['low']) / data['close']
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                   np.maximum(np.abs(data['high'] - data['close'].shift(1)), 
                                             np.abs(data['low'] - data['close'].shift(1))))
    data['vol_ratio'] = data['daily_range_vol'] / (np.abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)).replace(0, np.nan)
    data['vol_ratio'] = data['vol_ratio'].fillna(0)
    
    # Volatility-Scaled Acceleration
    data['range_adj_acc'] = data['net_acc_div'] / (data['daily_range_vol'] + 0.001)
    data['vol_stable_acc'] = data['net_acc_div'] * (1 / (data['vol_ratio'] + 0.001))
    data['vol_regime_acc'] = data['net_acc_div'] * data['daily_range_vol']
    
    # Multi-Scale Liquidity Pressure Analysis
    # Short-term Liquidity Dynamics (3-day)
    data['volume_intensity'] = data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean()
    data['liquidity_momentum'] = (data['amount'] / data['volume']) - (data['amount'].shift(3) / data['volume'].shift(3))
    data['directional_volume_intensity'] = data['volume'] * np.sign(data['close'] - data['close'].shift(1))
    
    # Medium-term Liquidity Asymmetry (8-day)
    # Volume Asymmetry Ratio
    def calc_volume_asymmetry(df_window):
        up_volume = df_window[df_window['close'] > df_window['close'].shift(1)]['volume'].sum()
        down_volume = df_window[df_window['close'] < df_window['close'].shift(1)]['volume'].sum()
        return up_volume / (down_volume + 0.001)
    
    volume_asymmetry = []
    for i in range(len(data)):
        if i >= 7:
            window_data = data.iloc[i-7:i+1].copy()
            window_data['prev_close'] = window_data['close'].shift(1)
            asym_ratio = calc_volume_asymmetry(window_data)
        else:
            asym_ratio = 1.0
        volume_asymmetry.append(asym_ratio)
    
    data['volume_asymmetry_ratio'] = volume_asymmetry
    
    # Liquidity Persistence
    data['liquidity_persistence'] = data['volume'].rolling(window=8).corr(data['close'])
    
    # Amount Efficiency Momentum
    data['amount_efficiency_momentum'] = ((data['amount'] / data['volume']) / (data['amount'].shift(8) / data['volume'].shift(8))) - 1
    
    # Liquidity-Divergence Integration
    data['volume_acc_alignment'] = data['net_acc_div'] * data['volume_intensity']
    data['liquidity_pressure_conf'] = data['range_adj_acc'] * data['liquidity_momentum']
    data['multi_scale_liquidity_consistency'] = data['volume_asymmetry_ratio'] * data['amount_efficiency_momentum']
    
    # Asymmetry Detection System
    # Directional Asymmetry
    data['pos_acc_strength'] = np.where(data['net_acc_div'] > 0, data['net_acc_div'], 0)
    data['neg_acc_strength'] = np.where(data['net_acc_div'] < 0, data['net_acc_div'], 0)
    data['asymmetry_ratio'] = data['pos_acc_strength'] / (np.abs(data['neg_acc_strength']) + 0.001)
    
    # Magnitude Asymmetry
    data['large_acc_detection'] = np.abs(data['net_acc_div']) > data['net_acc_div'].abs().rolling(window=10, min_periods=1).mean()
    data['small_acc_persistence'] = (np.abs(data['net_acc_div']) < data['net_acc_div'].abs().rolling(window=10, min_periods=1).mean()).rolling(window=5).sum()
    data['asymmetry_persistence'] = (data['asymmetry_ratio'] > 1).rolling(window=5).sum() - (data['asymmetry_ratio'] < 1).rolling(window=5).sum()
    
    # Gap-Pressure Acceleration Integration
    # Multi-Scale Gap Analysis
    # Short-term Gap Dynamics (3-day)
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_movement'] = (data['close'] - data['open']) / data['open']
    data['gap_fill_efficiency'] = data['intraday_movement'] / (data['overnight_gap'] + 0.001 * np.sign(data['overnight_gap']))
    
    # Medium-term Gap Compression (8-day)
    data['gap_compression_ratio'] = np.abs(data['open'] / data['close'].shift(1) - 1) / (np.abs(data['open'].shift(8) / data['close'].shift(9) - 1) + 0.001)
    data['gap_momentum'] = (data['close'] / data['close'].shift(8) - 1) * data['gap_fill_efficiency']
    
    # Gap-Acceleration Alignment
    data['gap_acc_correlation'] = data['net_acc_div'] * data['gap_fill_efficiency']
    data['multi_scale_gap_momentum'] = data['gap_momentum'] * data['range_adj_acc']
    data['gap_pressure_acc'] = data['vol_regime_acc'] * data['gap_compression_ratio']
    
    # Volume-Gap Fractal Confirmation
    data['volume_gap_alignment'] = data['directional_volume_intensity'] * data['gap_fill_efficiency']
    data['liquidity_pressure_gap_conf'] = data['liquidity_momentum'] * data['gap_compression_ratio']
    data['multi_scale_gap_liquidity_consistency'] = data['volume_asymmetry_ratio'] * data['gap_momentum']
    
    # Regime Classification Framework
    # Volatility Regime Detection
    data['current_true_range'] = data['true_range']
    data['avg_true_range_6d'] = data['true_range'].rolling(window=6, min_periods=1).mean()
    data['volatility_regime'] = data['current_true_range'] / (data['avg_true_range_6d'] + 0.001)
    
    # Acceleration Regime Classification
    net_acc_std_20 = data['net_acc_div'].rolling(window=20, min_periods=1).std()
    data['high_acc_regime'] = np.abs(data['net_acc_div']) > net_acc_std_20
    data['low_acc_regime'] = np.abs(data['net_acc_div']) < (0.5 * net_acc_std_20)
    data['normal_acc_regime'] = ~(data['high_acc_regime'] | data['low_acc_regime'])
    
    # Regime-Specific Factors
    # High Regime Components
    data['extreme_acc_reversal'] = data['net_acc_div'] / (net_acc_std_20 + 0.001)
    data['vol_amplified_acc'] = data['range_adj_acc'] * data['daily_range_vol']
    
    # Low Regime Components
    data['acc_accumulation'] = data['net_acc_div'].rolling(window=5, min_periods=1).sum()
    data['stability_enhanced_acc'] = data['net_acc_div'] / (data['net_acc_div'].rolling(window=10, min_periods=1).std() + 0.001)
    
    # Regime Transition Signals
    data['regime_change_detection'] = data['high_acc_regime'] != data['high_acc_regime'].shift(1)
    data['transition_momentum'] = data['net_acc_div'] * data['regime_change_detection']
    
    # Multi-Dimensional Signal Convergence
    # Signal Quality Assessment
    data['multi_scale_consistency'] = data['intraday_acc'].rolling(window=10).corr(data['short_term_acc'])
    data['liquidity_confirmation_strength'] = data['volume_acc_alignment'] * data['liquidity_pressure_conf']
    data['gap_pressure_alignment_quality'] = data['gap_acc_correlation'] * data['volume_gap_alignment']
    
    # Composite Acceleration Alpha Factor
    # Base Signal Construction
    data['acc_pressure_core'] = data['net_acc_div'] * data['liquidity_momentum']
    data['vol_enhanced_signal'] = data['acc_pressure_core'] * data['volatility_regime']
    data['gap_confirmed_signal'] = data['vol_enhanced_signal'] * data['gap_fill_efficiency']
    
    # Quality Adjustment
    consistency_weight = (data['multi_scale_consistency'] + 1) / 2  # Normalize to 0-1 range
    asymmetry_weight = np.tanh(data['asymmetry_ratio'])  # Bound asymmetry effect
    regime_confidence = np.where(data['regime_change_detection'], 0.5, 1.0)  # Reduce confidence during transitions
    
    # Final Alpha Output
    alpha_factor = (
        data['gap_confirmed_signal'] * 
        consistency_weight * 
        asymmetry_weight * 
        regime_confidence
    )
    
    return alpha_factor
