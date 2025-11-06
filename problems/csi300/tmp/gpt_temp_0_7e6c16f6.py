import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Asymmetry Analysis
    returns = data['close'] / data['close'].shift(1) - 1
    upside_returns = np.maximum(0, returns)
    downside_returns = np.maximum(0, -returns)
    
    upside_vol = upside_returns.rolling(window=10).std()
    downside_vol = downside_returns.rolling(window=10).std()
    volatility_asymmetry_ratio = upside_vol / downside_vol
    
    # Volume-Volatility Coupling
    volume_increase_mask = data['volume'] > data['volume'].shift(1)
    volume_decrease_mask = data['volume'] < data['volume'].shift(1)
    
    high_volume_vol = returns[volume_increase_mask].rolling(window=10).std()
    low_volume_vol = returns[volume_decrease_mask].rolling(window=10).std()
    volume_volatility_sensitivity = high_volume_vol / low_volume_vol
    
    # Asymmetry Persistence Patterns
    volatility_asymmetry_trend = np.sign(volatility_asymmetry_ratio - volatility_asymmetry_ratio.shift(5))
    volume_volatility_stability = volume_volatility_sensitivity.rolling(window=8).std()
    asymmetry_regime_shift = np.abs(volatility_asymmetry_ratio - volume_volatility_sensitivity)
    
    # Directional Price Momentum Structure
    micro_momentum = np.sign(data['close'] - data['close'].shift(3)) * np.abs(data['close'] / data['close'].shift(3) - 1)
    meso_momentum = np.sign(data['close'] - data['close'].shift(10)) * np.abs(data['close'] / data['close'].shift(10) - 1)
    macro_momentum = np.sign(data['close'] - data['close'].shift(25)) * np.abs(data['close'] / data['close'].shift(25) - 1)
    
    # Momentum Consistency Patterns
    momentum_signs = pd.DataFrame({
        'micro': np.sign(micro_momentum),
        'meso': np.sign(meso_momentum),
        'macro': np.sign(macro_momentum)
    })
    multi_scale_agreement = (momentum_signs > 0).sum(axis=1)
    
    micro_magnitude = np.abs(micro_momentum)
    meso_magnitude = np.abs(meso_momentum)
    momentum_magnitude_corr = micro_magnitude.rolling(window=10).corr(meso_magnitude)
    
    # Momentum Persistence Factor
    momentum_persistence = pd.Series(index=data.index, dtype=float)
    current_streak = 0
    prev_sign = 0
    for i, (idx, row) in enumerate(momentum_signs.iterrows()):
        if i == 0:
            momentum_persistence.iloc[i] = 0
            continue
        current_sign = np.sign(row.mean())
        if current_sign == prev_sign and current_sign != 0:
            current_streak += 1
        else:
            current_streak = 1 if current_sign != 0 else 0
        momentum_persistence.iloc[i] = current_streak
        prev_sign = current_sign
    
    # Regime-Dependent Momentum Characteristics
    high_asymmetry_momentum = micro_momentum * volatility_asymmetry_ratio
    low_asymmetry_momentum = micro_momentum / volatility_asymmetry_ratio
    volume_coupled_momentum = micro_momentum * volume_volatility_sensitivity
    
    # Amount-Price Efficiency Analysis
    price_move = np.abs(data['close'] - data['close'].shift(1))
    amount_per_price_move = data['amount'] / price_move.replace(0, np.nan)
    volume_per_price_move = data['volume'] / price_move.replace(0, np.nan)
    amount_volume_efficiency_ratio = amount_per_price_move / volume_per_price_move
    
    # Efficiency Trend Analysis
    efficiency_momentum = amount_volume_efficiency_ratio / amount_volume_efficiency_ratio.shift(5) - 1
    efficiency_volatility = amount_volume_efficiency_ratio.rolling(window=10).std()
    
    # Intraday Price Path Analysis
    high_low_range = data['high'] - data['low']
    opening_to_high_efficiency = (data['high'] - data['open']) / high_low_range.replace(0, np.nan)
    opening_to_low_efficiency = (data['open'] - data['low']) / high_low_range.replace(0, np.nan)
    close_to_extreme_efficiency = np.maximum(np.abs(data['close'] - data['high']), np.abs(data['close'] - data['low'])) / high_low_range.replace(0, np.nan)
    
    # Path Asymmetry Patterns
    upward_path_dominance = opening_to_high_efficiency - opening_to_low_efficiency
    downward_path_dominance = opening_to_low_efficiency - opening_to_high_efficiency
    close_location_asymmetry = np.abs(data['close'] - (data['high'] + data['low']) / 2) / high_low_range.replace(0, np.nan)
    
    # Volatility-Regime Adaptive Features
    volatility_regime = pd.Series('Balanced', index=data.index)
    volatility_regime[volatility_asymmetry_ratio > 1.5] = 'High'
    volatility_regime[volatility_asymmetry_ratio < 0.67] = 'Low'
    
    # Regime-Specific Momentum Enhancement
    high_asymmetry_features = micro_momentum * volume_volatility_sensitivity
    low_asymmetry_features = micro_momentum / volume_volatility_sensitivity
    balanced_features = micro_momentum * amount_volume_efficiency_ratio
    
    # Regime Transition Signals
    volatility_asymmetry_breakout = np.abs(volatility_asymmetry_ratio.diff()) > 0.3
    volume_volatility_decoupling = np.abs(volume_volatility_sensitivity.diff()) > 0.4
    
    # Regime Persistence Factor
    regime_persistence = pd.Series(index=data.index, dtype=int)
    current_regime_days = 0
    prev_regime = None
    for i, (idx, regime) in enumerate(volatility_regime.items()):
        if i == 0:
            regime_persistence.iloc[i] = 1
            prev_regime = regime
            continue
        if regime == prev_regime:
            current_regime_days += 1
        else:
            current_regime_days = 1
        regime_persistence.iloc[i] = current_regime_days
        prev_regime = regime
    
    # Composite Asymmetry Alpha Construction
    volatility_asymmetry_signal = volatility_asymmetry_ratio * volume_volatility_sensitivity
    directional_momentum_signal = multi_scale_agreement * momentum_persistence
    efficiency_asymmetry_signal = amount_volume_efficiency_ratio * upward_path_dominance
    
    # Regime-Adaptive Signal Combination
    high_asymmetry_composite = volatility_asymmetry_signal * high_asymmetry_features
    low_asymmetry_composite = efficiency_asymmetry_signal * low_asymmetry_features
    balanced_composite = directional_momentum_signal * balanced_features
    
    # Dynamic Signal Processing
    final_alpha = pd.Series(index=data.index, dtype=float)
    
    for i, (idx, regime) in enumerate(volatility_regime.items()):
        if regime == 'High':
            signal = high_asymmetry_composite.iloc[i]
        elif regime == 'Low':
            signal = low_asymmetry_composite.iloc[i]
        else:  # Balanced
            signal = balanced_composite.iloc[i]
        
        # Apply persistence enhancement
        signal *= regime_persistence.iloc[i]
        
        # Apply path direction confirmation
        path_direction = np.sign(upward_path_dominance.iloc[i])
        momentum_direction = np.sign(micro_momentum.iloc[i])
        if path_direction * momentum_direction > 0:
            signal *= 1.2  # Boost confirmed signals
        elif path_direction * momentum_direction < 0:
            signal *= 0.8  # Reduce conflicting signals
        
        final_alpha.iloc[i] = signal
    
    # Advanced Feature Refinement
    # Volatility Consistency Filter
    volatility_consistency_mask = regime_persistence >= 5
    # Momentum Confirmation Filter
    momentum_confirmation_mask = multi_scale_agreement >= 2
    # Efficiency Stability Filter
    efficiency_stability_mask = efficiency_volatility < efficiency_volatility.rolling(window=20).quantile(0.8)
    
    # Apply filters
    combined_mask = volatility_consistency_mask & momentum_confirmation_mask & efficiency_stability_mask
    final_alpha[~combined_mask] = final_alpha[~combined_mask] * 0.5  # Reduce weight for filtered signals
    
    # Persistence-Smoothed Output
    final_alpha_smoothed = final_alpha.rolling(window=5, min_periods=1).mean()
    
    # Regime-Adaptive Risk Scaling
    risk_scaling = pd.Series(1.0, index=data.index)
    risk_scaling[volatility_regime == 'High'] = 0.7
    risk_scaling[volatility_regime == 'Low'] = 1.3
    
    final_alpha_scaled = final_alpha_smoothed * risk_scaling
    
    return final_alpha_scaled
