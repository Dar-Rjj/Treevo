import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function for safe division
    def safe_divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    
    # Fractal Order Flow Imbalance
    high_low_range = data['high'] - data['low']
    data['fractal_upside_pressure'] = safe_divide(data['close'] - data['low'], high_low_range) * data['volume'] * safe_divide(data['amount'], data['amount'].shift(1))
    data['fractal_downside_pressure'] = safe_divide(data['high'] - data['close'], high_low_range) * data['volume'] * safe_divide(data['amount'], data['amount'].shift(1))
    data['net_fractal_order_flow'] = data['fractal_upside_pressure'] - data['fractal_downside_pressure']
    data['fractal_flow_momentum'] = data['net_fractal_order_flow'] - data['net_fractal_order_flow'].shift(1)
    
    # Multi-Timeframe Fractal Divergence
    data['immediate_fractal_shift'] = data['net_fractal_order_flow'] - data['net_fractal_order_flow'].shift(1)
    data['medium_term_fractal_gap'] = data['net_fractal_order_flow'] - data['net_fractal_order_flow'].shift(5)
    data['fractal_flow_acceleration'] = data['immediate_fractal_shift'] - (data['net_fractal_order_flow'].shift(1) - data['net_fractal_order_flow'].shift(2))
    data['fractal_flow_consistency'] = np.sign(data['immediate_fractal_shift']) * np.sign(data['medium_term_fractal_gap'])
    
    # Fractal Amount-Flow Dynamics
    data['fractal_amount_flow_momentum'] = data['amount'] * data['fractal_flow_momentum'] - data['amount'].shift(1) * data['fractal_flow_momentum'].shift(1)
    data['fractal_flow_concentration'] = safe_divide(data['amount'] * data['net_fractal_order_flow'], data['amount'].shift(1)) - safe_divide(data['amount'].shift(1) * data['net_fractal_order_flow'].shift(1), data['amount'].shift(2))
    data['fractal_high_value_flow'] = (data['amount'] > 2 * data['amount'].shift(1)) & (data['amount'] > data['amount'].shift(2))
    data['fractal_flow_value_alignment'] = np.sign(data['fractal_flow_momentum']) * np.sign(data['amount'] - data['amount'].shift(1))
    
    # Fractal Trading Intensity
    data['fractal_volume_intensity'] = safe_divide(data['volume'], data['volume'].shift(1)) - safe_divide(data['volume'].shift(1), data['volume'].shift(2))
    data['fractal_amount_intensity'] = safe_divide(data['amount'], data['amount'].shift(1)) - safe_divide(data['amount'].shift(1), data['amount'].shift(2))
    data['fractal_high_frequency_regime'] = (data['volume'] > 1.5 * data['volume'].shift(1)) & (data['amount'] > 1.5 * data['amount'].shift(1))
    data['fractal_low_liquidity_regime'] = (data['volume'] < 0.7 * data['volume'].shift(1)) & (data['amount'] < 0.7 * data['amount'].shift(1))
    
    # Fractal Price Discovery
    data['fractal_efficiency_measure'] = safe_divide(np.abs(data['close'] - data['open']), high_low_range) * safe_divide(data['amount'], data['amount'].shift(1))
    data['fractal_efficiency_shift'] = data['fractal_efficiency_measure'] - data['fractal_efficiency_measure'].shift(1)
    data['fractal_high_discovery'] = (data['fractal_efficiency_measure'] > 0.8) & (data['volume'] > data['volume'].shift(1))
    data['fractal_low_discovery'] = (data['fractal_efficiency_measure'] < 0.3) & (data['volume'] < data['volume'].shift(1))
    
    # Fractal Microstructure Alignment
    data['fractal_volume_amount_correlation'] = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['amount'] - data['amount'].shift(1))
    data['fractal_flow_efficiency_alignment'] = np.sign(data['fractal_flow_momentum']) * np.sign(data['fractal_efficiency_shift'])
    data['fractal_intensity_discovery_divergence'] = np.sign(data['fractal_volume_intensity']) * np.sign(data['fractal_efficiency_shift'])
    data['fractal_microstructure_consistency'] = data['fractal_volume_amount_correlation'] * data['fractal_flow_efficiency_alignment']
    
    # Fractal Flow Momentum Components
    data['fractal_upside_momentum'] = safe_divide((data['close'] - data['open']) * data['fractal_upside_pressure'], high_low_range)
    data['fractal_downside_momentum'] = safe_divide((data['open'] - data['close']) * data['fractal_downside_pressure'], high_low_range)
    data['net_fractal_momentum'] = data['fractal_upside_momentum'] - data['fractal_downside_momentum']
    data['fractal_momentum_acceleration'] = data['net_fractal_momentum'] - data['net_fractal_momentum'].shift(1)
    
    # Multi-Scale Fractal Momentum
    data['short_term_fractal_momentum'] = safe_divide(data['net_fractal_momentum'], np.abs(data['close'] - data['open']))
    data['medium_term_fractal_momentum'] = safe_divide(
        data['net_fractal_momentum'] + data['net_fractal_momentum'].shift(1) + data['net_fractal_momentum'].shift(2),
        np.abs(data['close'] - data['open']) + np.abs(data['close'].shift(1) - data['open'].shift(1)) + np.abs(data['close'].shift(2) - data['open'].shift(2))
    )
    data['fractal_momentum_gap'] = data['short_term_fractal_momentum'] - data['medium_term_fractal_momentum']
    data['fractal_momentum_divergence'] = np.sign(data['fractal_momentum_gap']) * np.sign(data['fractal_momentum_acceleration'])
    
    # Fractal Regime-Enhanced Momentum
    data['fractal_high_frequency_momentum'] = data['fractal_high_frequency_regime'].astype(float) * data['fractal_momentum_acceleration']
    data['fractal_discovery_momentum'] = data['fractal_high_discovery'].astype(float) * data['net_fractal_momentum']
    data['fractal_low_liquidity_momentum'] = data['fractal_low_liquidity_regime'].astype(float) * data['fractal_momentum_gap']
    data['fractal_recovery_momentum'] = data['fractal_low_liquidity_regime'].astype(float) * data['fractal_high_discovery'].astype(float) * data['fractal_momentum_acceleration']
    
    # Fractal Distribution Asymmetry
    data['fractal_price_position'] = safe_divide(data['close'] - (data['high'] + data['low'])/2, high_low_range) * safe_divide(data['amount'], data['amount'].shift(1))
    data['fractal_volume_distribution'] = safe_divide(data['fractal_upside_pressure'] - data['fractal_downside_pressure'], data['volume'])
    data['fractal_combined_skew'] = data['fractal_price_position'] * data['fractal_volume_distribution']
    data['fractal_skew_momentum'] = data['fractal_combined_skew'] - data['fractal_combined_skew'].shift(1)
    
    # Multi-Timeframe Fractal Skew
    data['fractal_immediate_skew'] = data['fractal_combined_skew'] - data['fractal_combined_skew'].shift(1)
    data['fractal_medium_term_skew'] = data['fractal_combined_skew'] - data['fractal_combined_skew'].shift(5)
    data['fractal_skew_acceleration'] = data['fractal_immediate_skew'] - (data['fractal_combined_skew'].shift(1) - data['fractal_combined_skew'].shift(2))
    data['fractal_skew_consistency'] = np.sign(data['fractal_immediate_skew']) * np.sign(data['fractal_medium_term_skew'])
    
    # Fractal Regime Distribution
    data['fractal_high_frequency_skew'] = data['fractal_high_frequency_regime'].astype(float) * data['fractal_combined_skew']
    data['fractal_discovery_distribution'] = data['fractal_efficiency_shift'] * data['fractal_volume_distribution']
    data['fractal_low_liquidity_distribution'] = data['fractal_low_liquidity_regime'].astype(float) * data['fractal_price_position']
    data['fractal_recovery_distribution'] = data['fractal_low_liquidity_regime'].astype(float) * data['fractal_high_discovery'].astype(float) * data['fractal_combined_skew']
    
    # Fractal Persistence Validation
    def calculate_persistence(series, window=3):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) >= 2:
                consistency_count = sum(np.sign(window_data.iloc[j]) == np.sign(window_data.iloc[j-1]) for j in range(1, len(window_data)))
                result.iloc[i] = consistency_count / (len(window_data) - 1)
        return result
    
    data['fractal_flow_consistency_persistence'] = calculate_persistence(data['net_fractal_order_flow'])
    data['fractal_momentum_persistence'] = calculate_persistence(data['fractal_flow_momentum'])
    data['fractal_efficiency_persistence'] = calculate_persistence(data['fractal_efficiency_measure'])
    data['fractal_discovery_consistency'] = calculate_persistence(data['fractal_efficiency_shift'])
    
    def calculate_regime_stability(series, window=3):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            window_data = series.iloc[i-window+1:i+1]
            result.iloc[i] = window_data.sum() / len(window_data)
        return result
    
    data['fractal_high_frequency_stability'] = calculate_regime_stability(data['fractal_high_frequency_regime'])
    data['fractal_discovery_stability'] = calculate_regime_stability(data['fractal_high_discovery'])
    
    # Core Fractal Components
    data['fractal_flow_regime'] = data['fractal_amount_flow_momentum'] * data['fractal_volume_intensity']
    data['fractal_momentum_regime'] = data['fractal_momentum_acceleration'] * data['fractal_efficiency_shift']
    data['fractal_skew_regime'] = data['fractal_skew_momentum'] * data['fractal_skew_acceleration']
    data['fractal_discovery_regime'] = data['fractal_efficiency_shift'] * data['fractal_flow_momentum']
    
    # Fractal-Confirmed Signals
    data['fractal_flow_intensity'] = data['fractal_flow_regime'] * data['fractal_volume_intensity']
    data['fractal_momentum_discovery'] = data['fractal_momentum_regime'] * data['fractal_momentum_divergence']
    data['fractal_skew_consistency_signal'] = data['fractal_skew_regime'] * data['fractal_skew_consistency']
    data['fractal_discovery_flow'] = data['fractal_discovery_regime'] * data['fractal_flow_efficiency_alignment']
    
    # Final Fractal Alpha Components
    data['primary_fractal_factor'] = data['fractal_flow_intensity'] * data['net_fractal_order_flow']
    data['secondary_fractal_factor'] = data['fractal_momentum_discovery'] * data['fractal_flow_consistency']
    data['tertiary_fractal_factor'] = data['fractal_skew_consistency_signal'] * data['fractal_volume_distribution']
    data['quaternary_fractal_factor'] = data['fractal_discovery_flow'] * data['fractal_price_position']
    
    # Composite Fractal Alpha with Persistence Multipliers
    flow_persistence_multiplier = 1 + data['fractal_flow_consistency_persistence']
    efficiency_persistence_multiplier = 1 + data['fractal_efficiency_persistence']
    regime_stability_multiplier = 1 + data['fractal_high_frequency_stability']
    discovery_stability_multiplier = 1 + data['fractal_discovery_stability']
    
    composite_alpha = (
        data['primary_fractal_factor'] * flow_persistence_multiplier +
        data['secondary_fractal_factor'] * efficiency_persistence_multiplier +
        data['tertiary_fractal_factor'] * regime_stability_multiplier +
        data['quaternary_fractal_factor'] * discovery_stability_multiplier
    )
    
    # Fill NaN values with 0
    composite_alpha = composite_alpha.fillna(0)
    
    return composite_alpha
