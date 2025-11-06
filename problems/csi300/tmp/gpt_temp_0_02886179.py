import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate ATR
    def calculate_atr(data, window):
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    # Calculate VWAP
    def calculate_vwap(data):
        return (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    
    # Calculate moving averages
    data['amount_ma5'] = data['amount'].rolling(window=5).mean()
    data['amount_ma20'] = data['amount'].rolling(window=20).mean()
    
    # Calculate ATRs
    data['atr3'] = calculate_atr(data, 3)
    data['atr5'] = calculate_atr(data, 5)
    data['atr10'] = calculate_atr(data, 10)
    
    # Calculate VWAP
    data['vwap'] = calculate_vwap(data)
    
    # 1. Asymmetric Volatility Regime Detection
    data['vol_clustering_intensity'] = data['atr3'] / data['atr10']
    data['upside_vol_efficiency'] = (data['high'] - data['close']) / np.abs(data['close'] - data['open'])
    data['downside_vol_efficiency'] = (data['close'] - data['low']) / np.abs(data['close'] - data['open'])
    
    # Volatility regime classification
    vol_regime_conditions = [
        (data['vol_clustering_intensity'] > 1.2) & (data['upside_vol_efficiency'] > data['downside_vol_efficiency']),
        (data['vol_clustering_intensity'] < 0.8) & (data['upside_vol_efficiency'] < data['downside_vol_efficiency']),
        (data['vol_clustering_intensity'] > 1.5)
    ]
    vol_regime_choices = ['high_vol_up', 'high_vol_down', 'extreme_vol']
    data['vol_regime'] = np.select(vol_regime_conditions, vol_regime_choices, default='normal_vol')
    
    # 2. Multi-Horizon Flow-Gap Integration
    data['short_flow_gap'] = ((data['amount'] / data['amount'].shift(1) - 
                              data['volume'] / data['volume'].shift(1)) * 
                             (data['close'] - data['open']) / data['atr3'])
    
    data['amount_change_5d'] = data['amount'] / data['amount'].shift(5)
    data['volume_change_5d'] = data['volume'] / data['volume'].shift(5)
    data['medium_flow_gap'] = ((data['amount_change_5d'] / data['volume_change_5d']) * 
                              (data['close'] - data['close'].shift(5)) / data['atr10'])
    
    # Flow-gap convergence
    data['flow_gap_convergence'] = data['short_flow_gap'].rolling(window=5).corr(data['medium_flow_gap'])
    
    # Regime-weighted flow-gap composite
    regime_weights = {
        'high_vol_up': 0.6, 'high_vol_down': 0.6, 'extreme_vol': 0.3, 'normal_vol': 0.8
    }
    data['regime_weight'] = data['vol_regime'].map(regime_weights)
    data['flow_gap_composite'] = (data['regime_weight'] * 
                                 (0.6 * data['short_flow_gap'] + 0.4 * data['medium_flow_gap']))
    
    # 3. Volume-Price Fractal Anchoring
    # Micro-fractal anchoring
    data['price_range_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['vwap_gap'] = (data['close'] - data['vwap']) / data['close']
    data['micro_fractal'] = data['price_range_efficiency'] * data['volume_concentration'] * np.abs(data['vwap_gap'])
    
    # Macro-fractal persistence
    data['price_efficiency_5d'] = (data['close'] - data['close'].shift(5)).abs() / (
        data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    data['volume_persistence_5d'] = data['volume'].rolling(window=5).std() / data['volume'].rolling(window=5).mean()
    data['gap_anchoring_consistency'] = (data['close'] - data['open']).rolling(window=5).std() / (
        data['close'] - data['open']).rolling(window=5).mean().abs()
    data['macro_fractal'] = data['price_efficiency_5d'] * data['volume_persistence_5d'] * data['gap_anchoring_consistency']
    
    # Fractal dimension mismatch
    data['fractal_mismatch'] = data['micro_fractal'] / (data['macro_fractal'] + 1e-8)
    data['fractal_adjusted_momentum'] = data['flow_gap_composite'] * (1 + 0.2 * data['fractal_mismatch'])
    
    # 4. Liquidity-Gap Compression Dynamics
    data['price_compression_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['volume_range_20d'] = data['volume'].rolling(window=20).max() - data['volume'].rolling(window=20).min()
    data['volume_compression_ratio'] = data['volume'] / (data['volume_range_20d'] + 1e-8)
    data['liquidity_gap_intensity'] = (data['volume'] / (data['high'] - data['low'] + 1e-8)) / (
        data['volume'] / (np.abs(data['close'] - data['close'].shift(1)) + 1e-8))
    
    # Compression breakout signal
    data['compression_breakout'] = (
        (data['price_compression_efficiency'] > 0.8) & 
        (data['volume_compression_ratio'] > 0.7) & 
        (data['liquidity_gap_intensity'] > 1)
    ).astype(int)
    
    # 5. Multi-Frequency Gap Entanglement
    def calculate_gap_correlation(data, window):
        close_open_gaps = np.abs(data['close'] - data['open'])
        high_low_ranges = data['high'] - data['low']
        correlations = []
        for i in range(len(data)):
            if i >= window - 1:
                corr = close_open_gaps.iloc[i-window+1:i+1].corr(high_low_ranges.iloc[i-window+1:i+1])
                correlations.append(corr)
            else:
                correlations.append(np.nan)
        return pd.Series(correlations, index=data.index)
    
    data['short_gap_correlation'] = calculate_gap_correlation(data, 3)
    data['medium_gap_correlation'] = calculate_gap_correlation(data, 7)
    data['gap_entanglement_strength'] = data['short_gap_correlation'] / (data['medium_gap_correlation'] + 1e-8)
    
    # Entanglement regime classification
    entanglement_conditions = [
        data['gap_entanglement_strength'] > 1.5,
        data['gap_entanglement_strength'] < 0.5
    ]
    entanglement_choices = ['high_entanglement', 'low_entanglement']
    data['entanglement_regime'] = np.select(entanglement_conditions, entanglement_choices, default='normal_entanglement')
    
    # 6. Flow-Volume Alignment Dynamics
    data['short_alignment'] = np.sign(data['close'] - data['open']) * np.sign(data['amount'] - data['amount'].shift(1))
    data['medium_alignment'] = np.sign(data['close'] - data['close'].shift(5)) * np.sign(data['amount'] - data['amount_ma5'])
    data['long_alignment'] = np.sign(data['close'] - data['close'].shift(20)) * np.sign(data['amount'] - data['amount_ma20'])
    data['multi_scale_alignment'] = (0.4 * data['short_alignment'] + 
                                   0.35 * data['medium_alignment'] + 
                                   0.25 * data['long_alignment'])
    
    # 7. Noise-Adjusted Volatility Flow
    data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
    data['shadow_pressure_asymmetry'] = (data['upper_shadow'] - data['lower_shadow']) / (data['high'] - data['low'] + 1e-8)
    
    data['gap_noise_intensity'] = (np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * (
        1 - np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    
    data['flow_decoherence'] = np.sign(data['flow_gap_convergence'].diff()) * np.abs(
        data['atr5'] - data['atr10']) / (data['atr10'] + 1e-8)
    
    # 8. Regime-Adaptive Signal Integration
    # Component weights based on regimes
    def get_component_weights(vol_regime, entanglement_regime):
        base_weights = {
            'flow_gap': 0.25, 'fractal': 0.20, 'liquidity': 0.20, 
            'entanglement': 0.15, 'alignment': 0.20
        }
        
        if entanglement_regime == 'high_entanglement':
            return {
                'flow_gap': 0.15, 'fractal': 0.25, 'liquidity': 0.15, 
                'entanglement': 0.30, 'alignment': 0.15
            }
        elif entanglement_regime == 'low_entanglement':
            return {
                'flow_gap': 0.35, 'fractal': 0.10, 'liquidity': 0.25, 
                'entanglement': 0.10, 'alignment': 0.20
            }
        else:  # normal_entanglement
            if vol_regime in ['high_vol_up', 'high_vol_down', 'extreme_vol']:
                return {
                    'flow_gap': 0.20, 'fractal': 0.25, 'liquidity': 0.25, 
                    'entanglement': 0.15, 'alignment': 0.15
                }
            else:
                return base_weights
    
    # Calculate final composite with regime-adaptive weights
    final_alpha = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 20:  # Ensure enough data for calculations
            final_alpha.iloc[i] = 0
            continue
            
        vol_regime = data['vol_regime'].iloc[i]
        entanglement_regime = data['entanglement_regime'].iloc[i]
        weights = get_component_weights(vol_regime, entanglement_regime)
        
        component_flow_gap = data['fractal_adjusted_momentum'].iloc[i]
        component_fractal = data['micro_fractal'].iloc[i] * data['macro_fractal'].iloc[i]
        component_liquidity = data['liquidity_gap_intensity'].iloc[i] * data['compression_breakout'].iloc[i]
        component_entanglement = data['gap_entanglement_strength'].iloc[i]
        component_alignment = data['multi_scale_alignment'].iloc[i]
        
        composite = (
            weights['flow_gap'] * component_flow_gap +
            weights['fractal'] * component_fractal +
            weights['liquidity'] * component_liquidity +
            weights['entanglement'] * component_entanglement +
            weights['alignment'] * component_alignment
        )
        
        # Apply noise adjustment
        shadow_pressure = np.abs(data['shadow_pressure_asymmetry'].iloc[i])
        gap_noise = data['gap_noise_intensity'].iloc[i]
        flow_decoherence = np.abs(data['flow_decoherence'].iloc[i])
        
        noise_adjustment = 1 + shadow_pressure + gap_noise + flow_decoherence
        
        final_alpha.iloc[i] = composite / noise_adjustment
    
    # Clean and return the final alpha series
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    return final_alpha
