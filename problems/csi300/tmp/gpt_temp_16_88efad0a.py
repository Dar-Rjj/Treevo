import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Directional Gap-Pressure Asymmetry
    data['prev_close'] = data['close'].shift(1)
    data['bull_gap'] = np.maximum(0, (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8))
    data['bear_gap'] = np.maximum(0, (data['prev_close'] - data['open']) / (data['prev_close'] + 1e-8))
    data['gap_asymmetry_ratio'] = data['bull_gap'] / (data['bear_gap'] + 0.0001)
    
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['upward_pressure'] = np.maximum(0, data['mid_price'] - data['close'])
    data['downward_pressure'] = np.maximum(0, data['close'] - data['mid_price'])
    data['pressure_asymmetry'] = data['upward_pressure'] / (data['downward_pressure'] + 0.0001)
    
    data['bull_alignment'] = data['bull_gap'] * data['upward_pressure']
    data['bear_alignment'] = data['bear_gap'] * data['downward_pressure']
    data['net_directional_pressure'] = data['bull_alignment'] - data['bear_alignment']
    
    # Fractal Liquidity Regime Detection
    data['depth_efficiency'] = (data['close'] - data['open']) / (data['amount'] + 1e-8)
    
    # Depth persistence (correlation over 5-day window)
    depth_persistence = []
    for i in range(len(data)):
        if i >= 4:
            window_amount = data['amount'].iloc[i-4:i+1].values
            window_price_diff = (data['close'] - data['open']).iloc[i-4:i+1].values
            if len(window_amount) > 1 and len(window_price_diff) > 1:
                corr = np.corrcoef(window_amount, window_price_diff)[0,1]
                depth_persistence.append(corr if not np.isnan(corr) else 0)
            else:
                depth_persistence.append(0)
        else:
            depth_persistence.append(0)
    data['depth_persistence'] = depth_persistence
    data['depth_regime'] = np.sign(data['depth_efficiency']) * data['depth_persistence']
    
    # Volume Fractal Structure
    data['volume_momentum'] = data['volume'] / (data['volume'].shift(5) + 1e-8) - 1
    
    # Volume volatility (5-day window)
    volume_volatility = []
    for i in range(len(data)):
        if i >= 4:
            window_volume = data['volume'].iloc[i-4:i+1].values
            vol_std = np.std(window_volume)
            vol_mean = np.mean(window_volume)
            volume_volatility.append(vol_std / (vol_mean + 1e-8))
        else:
            volume_volatility.append(0)
    data['volume_volatility'] = volume_volatility
    data['volume_regime'] = data['volume_momentum'] / (data['volume_volatility'] + 0.0001)
    
    # Liquidity Quality Fractals
    data['vwap_ratio'] = data['amount'] / (data['volume'] + 1e-8)
    data['liquidity_momentum'] = data['vwap_ratio'] / (data['vwap_ratio'].shift(3) + 1e-8) - 1
    
    # Liquidity volatility (5-day window)
    liquidity_volatility = []
    for i in range(len(data)):
        if i >= 4:
            window_vwap = data['vwap_ratio'].iloc[i-4:i+1].values
            liq_std = np.std(window_vwap)
            liq_mean = np.mean(window_vwap)
            liquidity_volatility.append(liq_std / (liq_mean + 1e-8))
        else:
            liquidity_volatility.append(0)
    data['liquidity_volatility'] = liquidity_volatility
    data['quality_regime'] = data['liquidity_momentum'] / (data['liquidity_volatility'] + 0.0001)
    
    # Asymmetric Gap-Pressure Momentum
    data['gap_asymmetry_momentum'] = data['gap_asymmetry_ratio'] / (data['gap_asymmetry_ratio'].shift(3) + 0.0001)
    data['pressure_asymmetry_momentum'] = data['pressure_asymmetry'] / (data['pressure_asymmetry'].shift(3) + 0.0001)
    data['net_pressure_momentum'] = data['net_directional_pressure'] / (data['net_directional_pressure'].shift(3) + 0.0001)
    
    # Medium-term Asymmetry (8-day)
    bull_gap_sum = data['bull_gap'].rolling(window=8, min_periods=1).sum()
    bear_gap_sum = data['bear_gap'].rolling(window=8, min_periods=1).sum()
    data['gap_persistence'] = bull_gap_sum / (bear_gap_sum + 0.0001)
    
    upward_pressure_sum = data['upward_pressure'].rolling(window=8, min_periods=1).sum()
    downward_pressure_sum = data['downward_pressure'].rolling(window=8, min_periods=1).sum()
    data['pressure_persistence'] = upward_pressure_sum / (downward_pressure_sum + 0.0001)
    
    net_pressure_sum = data['net_directional_pressure'].rolling(window=8, min_periods=1).sum()
    abs_net_pressure_sum = data['net_directional_pressure'].abs().rolling(window=8, min_periods=1).sum()
    data['directional_persistence'] = net_pressure_sum / (abs_net_pressure_sum + 0.0001)
    
    # Long-term Asymmetry (20-day)
    bull_gap_mean = data['bull_gap'].rolling(window=20, min_periods=1).mean()
    bear_gap_mean = data['bear_gap'].rolling(window=20, min_periods=1).mean()
    data['structural_gap_bias'] = bull_gap_mean / (bear_gap_mean + 0.0001)
    
    upward_pressure_mean = data['upward_pressure'].rolling(window=20, min_periods=1).mean()
    downward_pressure_mean = data['downward_pressure'].rolling(window=20, min_periods=1).mean()
    data['structural_pressure_bias'] = upward_pressure_mean / (downward_pressure_mean + 0.0001)
    
    # Regime stability (20-day correlation)
    regime_stability = []
    for i in range(len(data)):
        if i >= 19:
            window_gap = data['gap_asymmetry_ratio'].iloc[i-19:i+1].values
            window_pressure = data['pressure_asymmetry'].iloc[i-19:i+1].values
            if len(window_gap) > 1 and len(window_pressure) > 1:
                corr = np.corrcoef(window_gap, window_pressure)[0,1]
                regime_stability.append(corr if not np.isnan(corr) else 0)
            else:
                regime_stability.append(0)
        else:
            regime_stability.append(0)
    data['regime_stability'] = regime_stability
    
    # Fractal Regime Integration
    data['depth_gap_alignment'] = data['depth_regime'] * data['gap_asymmetry_ratio']
    data['depth_pressure_alignment'] = data['depth_regime'] * data['pressure_asymmetry']
    data['depth_direction_alignment'] = data['depth_regime'] * data['net_directional_pressure']
    
    data['volume_gap_confirmation'] = data['volume_regime'] * data['gap_asymmetry_momentum']
    data['volume_pressure_confirmation'] = data['volume_regime'] * data['pressure_asymmetry_momentum']
    data['volume_direction_confirmation'] = data['volume_regime'] * data['directional_persistence']
    
    data['quality_gap_enhancement'] = data['quality_regime'] * data['structural_gap_bias']
    data['quality_pressure_enhancement'] = data['quality_regime'] * data['structural_pressure_bias']
    data['quality_stability_enhancement'] = data['quality_regime'] * data['regime_stability']
    
    # Asymmetric Momentum Convergence
    data['short_medium_asymmetry'] = data['gap_asymmetry_momentum'] / (data['gap_persistence'] + 0.0001)
    data['medium_long_asymmetry'] = data['pressure_persistence'] / (data['structural_pressure_bias'] + 0.0001)
    data['multi_scale_asymmetry'] = data['short_medium_asymmetry'] * data['medium_long_asymmetry']
    
    data['depth_confirmed_momentum'] = data['depth_direction_alignment'] * data['directional_persistence']
    data['volume_confirmed_momentum'] = data['volume_direction_confirmation'] * data['net_pressure_momentum']
    data['quality_confirmed_momentum'] = data['quality_stability_enhancement'] * data['regime_stability']
    
    # Regime coherence (simplified)
    data['regime_coherence'] = data['depth_regime'] * data['volume_regime'] * data['quality_regime']
    
    # Momentum coherence (simplified)
    data['momentum_coherence'] = data['gap_asymmetry_momentum'] * data['pressure_asymmetry_momentum'] * data['net_pressure_momentum']
    
    data['signal_strength'] = data['regime_coherence'] * data['momentum_coherence']
    
    # Dynamic Signal Construction
    data['base_asymmetry'] = data['net_directional_pressure'] * data['gap_asymmetry_ratio']
    data['enhanced_asymmetry'] = data['base_asymmetry'] * data['pressure_asymmetry']
    data['core_signal'] = data['enhanced_asymmetry'] * data['multi_scale_asymmetry']
    
    data['depth_weighted_signal'] = data['core_signal'] * data['depth_direction_alignment']
    data['volume_weighted_signal'] = data['depth_weighted_signal'] * data['volume_direction_confirmation']
    data['quality_weighted_signal'] = data['volume_weighted_signal'] * data['quality_stability_enhancement']
    
    data['short_term_validation'] = data['quality_weighted_signal'] * data['net_pressure_momentum']
    data['medium_term_validation'] = data['short_term_validation'] * data['directional_persistence']
    data['long_term_validation'] = data['medium_term_validation'] * data['regime_stability']
    
    # Composite Asymmetric Alpha
    data['regime_consistency'] = np.abs(data['depth_regime'] * data['volume_regime'] * data['quality_regime'])
    data['momentum_consistency'] = np.abs(data['gap_asymmetry_momentum'] * data['pressure_asymmetry_momentum'] * data['net_pressure_momentum'])
    data['signal_filter'] = data['regime_consistency'] * data['momentum_consistency']
    
    data['regime_amplification'] = data['long_term_validation'] * data['regime_coherence']
    data['momentum_amplification'] = data['regime_amplification'] * data['momentum_coherence']
    data['quality_amplification'] = data['momentum_amplification'] * data['signal_strength']
    
    data['filtered_signal'] = data['quality_amplification'] * data['signal_filter']
    data['volatility_adjustment'] = data['filtered_signal'] / (1 + data['volume_volatility'])
    data['alpha_value'] = data['volatility_adjustment'] * data['regime_stability']
    
    return data['alpha_value']
