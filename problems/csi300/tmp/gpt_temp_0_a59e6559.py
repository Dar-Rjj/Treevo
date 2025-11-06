import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Fractured Overnight Momentum Detection
    df['overnight_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['overnight_fracture_gap'] = df['overnight_return'] - df['intraday_return']
    
    # Multi-Day Fracture Persistence
    df['fracture_magnitude'] = np.abs(df['overnight_fracture_gap'])
    fracture_streak = []
    current_streak = 0
    for i, mag in enumerate(df['fracture_magnitude']):
        if mag > 0.005:
            current_streak += 1
        else:
            current_streak = 0
        fracture_streak.append(current_streak)
    df['fracture_streak'] = fracture_streak
    df['fracture_persistence_score'] = df['fracture_magnitude'] * df['fracture_streak']
    
    # Volume Concentration Divergence
    daily_range = df['high'] - df['low']
    high_range_threshold = df['low'] + 0.8 * daily_range
    low_range_threshold = df['low'] + 0.2 * daily_range
    
    high_range_volume = df['volume'] * ((df['close'] >= high_range_threshold) | (df['open'] >= high_range_threshold)).astype(int)
    low_range_volume = df['volume'] * ((df['close'] <= low_range_threshold) | (df['open'] <= low_range_threshold)).astype(int)
    
    df['volume_concentration_ratio'] = high_range_volume / (low_range_volume + 1e-6)
    df['concentration_divergence'] = df['volume_concentration_ratio'] * df['overnight_fracture_gap']
    
    # Liquidity Stress Divergence
    df['amount_per_trade'] = df['amount'] / (df['volume'] + 1e-6)
    df['amount_volatility'] = np.abs(df['amount'] - df['amount'].shift(1)) / (df['amount'].shift(1) + 1e-6)
    df['liquidity_stress'] = df['amount_volatility'] * df['amount_per_trade']
    df['stress_divergence'] = df['liquidity_stress'] * df['overnight_fracture_gap']
    
    # Volume-Price Divergence Integration
    df['price_volume_divergence'] = (df['close'] - df['open']) / (df['volume'] + 1e-6) - (df['close'].shift(1) - df['open'].shift(1)) / (df['volume'].shift(1) + 1e-6)
    
    divergence_persistence = []
    for i in range(len(df)):
        if i < 2:
            divergence_persistence.append(0)
        else:
            window = df['price_volume_divergence'].iloc[max(0, i-2):i+1]
            count = np.sum(np.abs(window) > 0.0001)
            divergence_persistence.append(count)
    df['divergence_persistence'] = divergence_persistence
    df['integrated_divergence'] = df['price_volume_divergence'] * df['divergence_persistence'] * df['volume_concentration_ratio']
    
    # Microstructure Rejection Analysis
    df['upper_shadow_rejection'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] + 1e-6)
    df['lower_shadow_rejection'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['low'] + 1e-6)
    df['total_rejection'] = df['upper_shadow_rejection'] + df['lower_shadow_rejection']
    
    df['spread_estimate'] = 2 * np.abs(df['close'] - (df['high'] + df['low'])/2) / ((df['high'] + df['low'])/2 + 1e-6)
    df['spread_intensity'] = df['spread_estimate'] * df['volume']
    df['microstructure_stress'] = df['total_rejection'] * df['spread_intensity']
    
    # Adaptive Multi-Scale Regime Recognition
    df['daily_range_volatility'] = (df['high'] - df['low']) / (df['open'] + 1e-6)
    df['volatility_ratio'] = df['daily_range_volatility'] / (df['daily_range_volatility'].shift(1) + 1e-6)
    
    regime_change = []
    for i in range(len(df)):
        if i < 4:
            regime_change.append(0)
        else:
            window = df['volatility_ratio'].iloc[max(0, i-4):i+1]
            count = np.sum((window > 1.5) | (window < 0.67))
            regime_change.append(count)
    df['regime_change'] = regime_change
    df['volatility_regime_score'] = df['volatility_ratio'] * df['regime_change']
    
    df['volume_surge'] = df['volume'] / (df['volume'].shift(1) + 1e-6)
    
    volume_persistence = []
    current_persistence = 0
    for i in range(len(df)):
        if i == 0:
            volume_persistence.append(0)
        else:
            if df['volume'].iloc[i] > df['volume'].iloc[i-1]:
                current_persistence += 1
            else:
                current_persistence = 0
            volume_persistence.append(current_persistence)
    df['volume_persistence'] = volume_persistence
    df['volume_regime_score'] = df['volume_surge'] * df['volume_persistence']
    
    multi_scale_conflict = []
    conflict_persistence = []
    current_conflict_streak = 0
    for i in range(len(df)):
        if i == 0:
            multi_scale_conflict.append(0)
            conflict_persistence.append(0)
        else:
            opening_sign = np.sign(df['open'].iloc[i] - df['close'].iloc[i-1])
            intraday_sign = np.sign(df['close'].iloc[i] - df['open'].iloc[i])
            conflict = int(opening_sign != intraday_sign)
            multi_scale_conflict.append(conflict)
            
            if conflict == 1:
                current_conflict_streak += 1
            else:
                current_conflict_streak = 0
            conflict_persistence.append(current_conflict_streak)
    
    df['opening_intraday_conflict'] = multi_scale_conflict
    df['conflict_persistence'] = conflict_persistence
    df['multi_scale_score'] = df['opening_intraday_conflict'] * df['conflict_persistence']
    
    # Composite Fracture-Divergence Alpha
    df['core_fracture_component'] = df['overnight_fracture_gap'] * df['fracture_persistence_score']
    df['divergence_enhancement'] = df['core_fracture_component'] * df['integrated_divergence'] * df['stress_divergence']
    df['microstructure_adjustment'] = df['divergence_enhancement'] * df['microstructure_stress']
    df['final_factor'] = df['microstructure_adjustment'] * df['volatility_regime_score'] * df['volume_regime_score'] * df['multi_scale_score']
    
    return df['final_factor']
