import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Basic price calculations
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Fractal Range Dynamics
    data['daily_range_fractal'] = (data['high'] - data['low']) / (abs(data['close'] - data['prev_close']) + 1e-8)
    data['gap_absorption_fractal'] = abs(data['open'] - data['prev_close']) / (data['high'] - data['low'] + 1e-8)
    
    # Range persistence (5-day average range)
    data['range_5d_avg'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['range_persistence'] = (data['high'] - data['low'] > data['range_5d_avg']).rolling(window=5).sum()
    
    # Short-term and medium-term fractals
    data['short_term_fractal'] = data['daily_range_fractal'].rolling(window=3).mean()
    data['medium_term_fractal'] = data['daily_range_fractal'].rolling(window=10).mean()
    data['fractal_transition'] = abs(data['short_term_fractal'] / (data['medium_term_fractal'] + 1e-8) - 1)
    
    # Price Position Asymmetry
    data['open_high_low_ratio'] = (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['upper_rejection'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'] + 1e-8)
    data['lower_rejection'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['net_rejection_asymmetry'] = data['upper_rejection'] - data['lower_rejection']
    
    # Temporal Price Patterns
    data['mid_point'] = (data['high'] + data['low']) / 2
    data['morning_asymmetry'] = abs(data['mid_point'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['afternoon_asymmetry'] = abs(data['close'] - data['mid_point']) / (data['high'] - data['low'] + 1e-8)
    data['full_session_bias'] = np.sign(data['close'] - data['open']) * np.sign(data['mid_point'] - data['open'])
    
    # Volume Dynamics
    data['prev_volume_3'] = data['volume'].shift(3)
    data['volume_acceleration'] = (data['volume'] / (data['prev_volume_3'] + 1e-8)) ** (1/3) - 1
    
    data['prev_amount'] = data['amount'].shift(1)
    data['amount_flow_velocity'] = data['amount'] / (data['prev_amount'] + 1e-8) - 1
    
    data['amount_diff_1'] = data['amount'] - data['prev_amount']
    data['amount_diff_2'] = data['prev_amount'] - data['amount'].shift(2)
    data['amount_persistence'] = np.sign(data['amount_diff_1']) * np.sign(data['amount_diff_2'])
    
    # Volume at Key Price Levels
    data['gap_volume_efficiency'] = data['volume'] / (abs(data['open'] - data['prev_close']) + 1e-8)
    data['volume_clustering_rejection'] = data['volume'] * data['net_rejection_asymmetry']
    
    # Volume-Range Efficiency
    data['high_fractal_efficiency'] = data['amount'] / (data['volume'] * data['true_range'] + 1e-8)
    data['low_fractal_efficiency'] = data['amount'] / (data['volume'] * (data['high'] - data['low']) + 1e-8)
    data['volume_efficiency_premium'] = data['high_fractal_efficiency'] - data['low_fractal_efficiency']
    
    # Multi-Timeframe Rejection Asymmetry
    data['close_3d_max'] = data['close'].rolling(window=3).max()
    data['close_3d_min'] = data['close'].rolling(window=3).min()
    data['rejection_asymmetry_3d'] = (data['high'] - data['close_3d_max']) - (data['close_3d_min'] - data['low'])
    
    data['close_10d_max'] = data['close'].rolling(window=10).max()
    data['close_10d_min'] = data['close'].rolling(window=10).min()
    data['rejection_asymmetry_10d'] = (data['high'] - data['close_10d_max']) - (data['close_10d_min'] - data['low'])
    
    data['fractal_scaled_rejection'] = data['net_rejection_asymmetry'] * data['fractal_transition']
    
    # Volume-Momentum Alignment
    data['volume_ma_3'] = data['volume'].rolling(window=3).mean()
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_trend_direction'] = np.sign(data['volume_ma_3'] - data['volume_ma_5'])
    
    data['price_direction'] = np.sign(data['close'] - data['prev_close'])
    data['range_momentum_alignment'] = data['price_direction'] * data['volume_trend_direction']
    data['volume_confirmation_strength'] = abs(data['amount_flow_velocity']) * data['range_momentum_alignment']
    
    # Consecutive Day Momentum Decay
    data['close_diff'] = data['close'] - data['prev_close']
    data['momentum_direction'] = np.sign(data['close_diff'])
    
    # Calculate consecutive same direction days
    momentum_persistence = []
    current_streak = 0
    prev_dir = 0
    
    for i, row in data.iterrows():
        if i == data.index[0]:
            momentum_persistence.append(0)
            prev_dir = row['momentum_direction']
            continue
            
        if row['momentum_direction'] == prev_dir and prev_dir != 0:
            current_streak += 1
        else:
            current_streak = 1 if row['momentum_direction'] != 0 else 0
            
        momentum_persistence.append(current_streak)
        prev_dir = row['momentum_direction']
    
    data['momentum_persistence'] = momentum_persistence
    data['momentum_decay_factor'] = 1 / (1 + data['momentum_persistence'])
    data['decay_adjusted_momentum'] = data['close_diff'] * data['momentum_decay_factor']
    
    # Fractal Regime Classification
    data['high_fractal_regime'] = data['short_term_fractal'] > data['medium_term_fractal']
    data['low_fractal_regime'] = data['short_term_fractal'] < data['medium_term_fractal']
    data['transition_regime'] = data['fractal_transition'] > data['fractal_transition'].rolling(window=20).quantile(0.7)
    
    # Regime-Specific Weights
    data['high_fractal_weight'] = data['range_persistence'] / 5
    
    # Range autocorrelation for low fractal weight
    range_autocorr = []
    for i in range(len(data)):
        if i < 5:
            range_autocorr.append(0)
        else:
            window = data['high'].iloc[i-5:i+1] - data['low'].iloc[i-5:i+1]
            if len(window) >= 2:
                corr = window.autocorr()
                range_autocorr.append(abs(corr) if not np.isnan(corr) else 0)
            else:
                range_autocorr.append(0)
    
    data['range_autocorrelation'] = range_autocorr
    data['low_fractal_weight'] = 1 - data['range_autocorrelation']
    data['transition_weight'] = data['fractal_transition']
    
    # Volume-Adaptive Scaling
    data['volume_acceleration_weight'] = abs(data['volume_acceleration'])
    data['amount_flow_weight'] = abs(data['amount_flow_velocity'])
    data['efficiency_premium_weight'] = data['volume_efficiency_premium']
    
    # Composite Factor Construction
    # Price-Volume Interaction Components
    data['primary_component'] = data['rejection_asymmetry_3d'] * data['volume_acceleration']
    data['secondary_component'] = data['range_momentum_alignment'] * data['full_session_bias']
    data['tertiary_component'] = data['volume_efficiency_premium'] * data['morning_asymmetry']
    
    # Momentum Confirmation Layer
    data['volume_confirmation'] = data['amount_persistence'] * data['volume_confirmation_strength']
    data['fractal_momentum_alignment'] = data['fractal_scaled_rejection'] * data['decay_adjusted_momentum']
    data['temporal_bias_confirmation'] = data['afternoon_asymmetry'] * data['volume_clustering_rejection']
    
    # Regime-weighted components
    regime_weights = np.where(
        data['high_fractal_regime'], data['high_fractal_weight'],
        np.where(
            data['low_fractal_regime'], data['low_fractal_weight'],
            data['transition_weight']
        )
    )
    
    # Final composite calculation
    price_volume_composite = (
        data['primary_component'] + 
        data['secondary_component'] + 
        data['tertiary_component']
    )
    
    momentum_composite = (
        data['volume_confirmation'] + 
        data['fractal_momentum_alignment'] + 
        data['temporal_bias_confirmation']
    )
    
    regime_weighted_composite = (price_volume_composite + momentum_composite) * regime_weights
    
    # Volume-adaptive scaling
    volume_scaling = (
        data['volume_acceleration_weight'] + 
        data['amount_flow_weight'] + 
        data['efficiency_premium_weight']
    ) / 3
    
    final_factor = regime_weighted_composite * volume_scaling
    
    # Clean up and return
    result = pd.Series(final_factor, index=data.index)
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result
