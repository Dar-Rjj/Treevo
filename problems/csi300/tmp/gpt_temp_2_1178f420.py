import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Components
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_divergence'] = data['momentum_5d'] - data['momentum_10d']
    
    # Volume Momentum Analysis
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_divergence'] = data['volume_momentum_5d'] - data['volume_momentum_10d']
    
    # Divergence Strength Measurement
    data['divergence_product'] = data['momentum_divergence'] * data['volume_divergence']
    data['abs_divergence_magnitude'] = abs(data['momentum_divergence']) * abs(data['volume_divergence'])
    data['directional_alignment'] = np.sign(data['momentum_divergence']) * np.sign(data['volume_divergence'])
    
    # Volatility Breakout Detection
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['atr_5d'] = data['true_range'].rolling(window=5).mean()
    data['volatility_breakout'] = data['true_range'] / data['atr_5d']
    
    # Price Acceleration Measurement
    data['return'] = data['close'].pct_change()
    data['return_acceleration'] = (data['return'] - data['return'].shift(1)) - (data['return'].shift(1) - data['return'].shift(2))
    data['acceleration_std_5d'] = data['return_acceleration'].rolling(window=5).std()
    data['normalized_acceleration'] = data['return_acceleration'] / data['acceleration_std_5d'].replace(0, np.nan)
    
    # Regime Classification System
    data['high_vol_regime'] = data['volatility_breakout'] > 1.5
    data['low_vol_regime'] = data['volatility_breakout'] < 0.7
    data['acceleration_regime'] = abs(data['normalized_acceleration']) > 2.0
    data['stable_regime'] = ~(data['high_vol_regime'] | data['low_vol_regime'] | data['acceleration_regime'])
    
    # Amount-Based Flow Confirmation
    data['amount_momentum'] = data['amount'] / data['amount'].shift(5) - 1
    data['amount_volume_ratio'] = data['amount'] / data['volume']
    data['amount_volume_ratio_change'] = data['amount_volume_ratio'] / data['amount_volume_ratio'].shift(1) - 1
    data['flow_concentration'] = data['amount'] / data['amount'].rolling(window=5).mean()
    
    # Flow-Price Synchronization
    data['amount_price_correlation'] = np.sign(data['amount_momentum']) * np.sign(data['momentum_5d'])
    data['flow_acceleration_alignment'] = np.sign(data['amount_momentum']) * np.sign(data['return_acceleration'])
    data['volume_amount_divergence'] = data['volume_divergence'] - data['amount_momentum']
    
    # Institutional Flow Detection
    data['avg_amount_10d'] = data['amount'].rolling(window=10).mean()
    data['large_order_indicator'] = data['amount'] > (2 * data['avg_amount_10d'])
    data['amount_above_threshold'] = data['amount'] > (1.5 * data['amount'].rolling(window=3).mean())
    data['sustained_flow'] = data['amount_above_threshold'].rolling(window=3).sum()
    
    # Flow persistence score
    data['positive_amount_momentum'] = data['amount_momentum'] > 0
    data['flow_persistence'] = 0
    for i in range(1, len(data)):
        if data['positive_amount_momentum'].iloc[i]:
            data['flow_persistence'].iloc[i] = data['flow_persistence'].iloc[i-1] + 1
        else:
            data['flow_persistence'].iloc[i] = 0
    
    # Multi-Timeframe Convergence Engine
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['volume_momentum_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['momentum_convergence'] = np.sign(data['momentum_3d']) * np.sign(data['momentum_10d'])
    data['volume_confirmation'] = np.sign(data['volume_momentum_3d']) * np.sign(data['volume_momentum_10d'])
    
    # Acceleration trend
    data['acceleration_trend_5d'] = data['return_acceleration'].rolling(window=5).mean()
    data['acceleration_consistency'] = np.sign(data['return_acceleration']) * np.sign(data['acceleration_trend_5d'])
    
    # Regime Transition Timing
    data['vol_breakout_cross_up'] = (data['volatility_breakout'] > 1.5) & (data['volatility_breakout'].shift(1) <= 1.5)
    data['vol_breakout_cross_down'] = (data['volatility_breakout'] < 0.7) & (data['volatility_breakout'].shift(1) >= 0.7)
    
    # Acceleration peaks (local maxima/minima)
    data['acceleration_peak'] = (
        (data['normalized_acceleration'] > data['normalized_acceleration'].shift(1)) & 
        (data['normalized_acceleration'] > data['normalized_acceleration'].shift(-1))
    ) | (
        (data['normalized_acceleration'] < data['normalized_acceleration'].shift(1)) & 
        (data['normalized_acceleration'] < data['normalized_acceleration'].shift(-1))
    )
    
    # Flow confirmation timing
    data['amount_surge'] = data['amount_momentum'] > data['amount_momentum'].rolling(window=10).quantile(0.8)
    data['flow_confirmation_timing'] = data['amount_surge'] & (data['vol_breakout_cross_up'] | data['vol_breakout_cross_down'])
    
    # Convergence Strength Scoring
    alignment_indicators = [data['momentum_convergence'], data['volume_confirmation'], data['acceleration_consistency']]
    data['timeframe_alignment'] = sum([ind.fillna(0) for ind in alignment_indicators])
    
    data['regime_transition_premium'] = 0
    data.loc[data['vol_breakout_cross_up'] | data['vol_breakout_cross_down'], 'regime_transition_premium'] = 1.5
    data.loc[data['flow_confirmation_timing'], 'regime_transition_premium'] = 2.0
    
    data['flow_confirmation_bonus'] = 0
    data.loc[data['large_order_indicator'], 'flow_confirmation_bonus'] = 1.2
    data.loc[data['sustained_flow'] >= 2, 'flow_confirmation_bonus'] = 1.5
    
    # Dynamic Factor Integration
    # Core Divergence Foundation
    base_factor = data['divergence_product'] * data['abs_divergence_magnitude'] * data['directional_alignment']
    
    # Regime Acceleration Multiplier
    regime_multiplier = 1.0
    regime_multiplier = np.where(data['high_vol_regime'], 1.8, regime_multiplier)
    regime_multiplier = np.where(data['acceleration_regime'] & (data['directional_alignment'] > 0), 2.2, regime_multiplier)
    regime_multiplier = np.where(data['low_vol_regime'], 0.6, regime_multiplier)
    
    # Flow-Based Validation
    flow_validation = data['amount_price_correlation'].fillna(0)
    flow_validation = np.where(data['large_order_indicator'], flow_validation * data['flow_concentration'], flow_validation)
    flow_validation = flow_validation * (1 + 0.1 * data['flow_persistence'])
    
    # Convergence Optimization
    convergence_boost = 1.0 + 0.3 * data['timeframe_alignment']
    convergence_boost = convergence_boost * (1 + 0.2 * data['regime_transition_premium'])
    convergence_boost = np.where(data['volume_amount_divergence'].abs() > 0.1, 0.8, convergence_boost)
    convergence_boost = convergence_boost * (1 + 0.1 * data['timeframe_alignment'])
    
    # Final Alpha Factor Generation
    final_factor = (
        base_factor * 
        regime_multiplier * 
        flow_validation * 
        convergence_boost * 
        (1 + data['flow_confirmation_bonus'])
    )
    
    return final_factor
