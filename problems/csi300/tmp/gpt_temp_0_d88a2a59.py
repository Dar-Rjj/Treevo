import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Acceleration Calculation
    # Price Acceleration
    data['price_ret_3d'] = data['close'].pct_change(3)
    data['price_ret_8d'] = data['close'].pct_change(8)
    data['price_ret_21d'] = data['close'].pct_change(21)
    
    data['price_accel_3d'] = data['price_ret_3d'].diff(3)
    data['price_accel_8d'] = data['price_ret_8d'].diff(8)
    data['price_accel_21d'] = data['price_ret_21d'].diff(21)
    
    # Volume Acceleration
    data['volume_ret_3d'] = data['volume'].pct_change(3)
    data['volume_ret_8d'] = data['volume'].pct_change(8)
    data['volume_ret_21d'] = data['volume'].pct_change(21)
    
    data['volume_accel_3d'] = data['volume_ret_3d'].diff(3)
    data['volume_accel_8d'] = data['volume_ret_8d'].diff(8)
    data['volume_accel_21d'] = data['volume_ret_21d'].diff(21)
    
    # Acceleration Divergence Detection
    timeframes = ['3d', '8d', '21d']
    divergence_signals = {}
    divergence_magnitudes = {}
    
    for tf in timeframes:
        price_accel = data[f'price_accel_{tf}']
        volume_accel = data[f'volume_accel_{tf}']
        
        # Divergence signals
        divergence_signals[f'{tf}'] = np.where(
            price_accel > volume_accel, 1,
            np.where(price_accel < volume_accel, -1, 0)
        )
        
        # Divergence magnitudes
        divergence_magnitudes[f'{tf}'] = np.abs(price_accel - volume_accel)
    
    # Volatility Regime Classification
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['short_term_vol'] = data['daily_range'].rolling(window=5).mean()
    data['medium_term_vol'] = data['close'].pct_change().rolling(window=10).std()
    data['long_term_vol'] = (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()) / data['close']
    
    data['avg_range_20d'] = data['daily_range'].rolling(window=20).mean()
    data['range_std_20d'] = data['daily_range'].rolling(window=20).std()
    
    # Regime assignment
    upper_bound = data['avg_range_20d'] + 0.5 * data['range_std_20d']
    lower_bound = data['avg_range_20d'] - 0.5 * data['range_std_20d']
    
    data['vol_regime'] = np.where(
        data['daily_range'] > upper_bound, 2,  # High volatility
        np.where(data['daily_range'] < lower_bound, 0, 1)  # Low volatility, Normal volatility
    )
    
    # Regime-Adaptive Signal Processing
    regime_weights = {}
    
    for tf in timeframes:
        # High volatility regime: emphasize short-term, reduce long-term
        if tf == '3d':
            high_vol_weight = 1.5
            low_vol_weight = 0.7
            normal_weight = 1.0
        elif tf == '8d':
            high_vol_weight = 1.0
            low_vol_weight = 0.8
            normal_weight = 1.0
        else:  # 21d
            high_vol_weight = 0.5
            low_vol_weight = 1.3
            normal_weight = 1.0
        
        regime_weights[tf] = np.where(
            data['vol_regime'] == 2, high_vol_weight,
            np.where(data['vol_regime'] == 0, low_vol_weight, normal_weight)
        )
    
    # Signal Strength Integration
    signal_strength = pd.Series(0.0, index=data.index)
    weighted_divergence = pd.Series(0.0, index=data.index)
    
    for tf in timeframes:
        signal = divergence_signals[tf]
        magnitude = divergence_magnitudes[tf]
        weight = regime_weights[tf]
        
        # Count consistent divergence direction
        consistent_count = pd.Series(0, index=data.index)
        for other_tf in timeframes:
            if other_tf != tf:
                consistent_count += (divergence_signals[other_tf] == signal)
        
        # Signal strength multiplier based on consistency
        consistency_multiplier = 1.0 + (consistent_count / 2.0)
        
        # Regime persistence multiplier
        regime_persistence = data['vol_regime'].rolling(window=5).apply(lambda x: len(set(x)) == 1, raw=False).fillna(0)
        persistence_multiplier = 1.0 + (0.3 * regime_persistence)
        
        # Combine components
        tf_contribution = signal * magnitude * weight * consistency_multiplier * persistence_multiplier
        weighted_divergence += tf_contribution
        
        # Update signal strength
        signal_strength += np.abs(signal) * weight
    
    # Final Factor Construction
    # Normalize signal strength to avoid extreme values
    signal_strength_normalized = signal_strength / (signal_strength.rolling(window=63, min_periods=1).mean() + 1e-8)
    
    # Combine weighted divergence with signal strength
    final_factor = weighted_divergence * (1.0 + 0.2 * signal_strength_normalized)
    
    # Apply regime-specific final adjustments
    regime_adjustment = np.where(
        data['vol_regime'] == 2, 1.2,  # High volatility: amplify signals
        np.where(data['vol_regime'] == 0, 0.8, 1.0)  # Low volatility: dampen signals
    )
    
    final_factor = final_factor * regime_adjustment
    
    # Clean and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    return final_factor
