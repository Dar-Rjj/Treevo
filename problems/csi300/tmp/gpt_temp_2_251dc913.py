import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Association Asymmetry Momentum Factor
    Combines price-volume asymmetry, microstructure efficiency, and flow dynamics
    to generate regime-aware momentum signals.
    """
    data = df.copy()
    
    # 1. Multi-Scale Asymmetry Detection
    # Price-Volume Asymmetry Framework
    data['price_change'] = data['close'] - data['open']
    data['abs_price_change'] = abs(data['price_change'])
    data['day_direction'] = np.where(data['price_change'] > 0, 1, -1)
    
    # Directional volume intensity
    up_days = data['day_direction'] == 1
    down_days = data['day_direction'] == -1
    
    # Rolling volume asymmetry (5-day window)
    data['up_day_volume_ma'] = data['volume'].rolling(window=5).apply(
        lambda x: x[up_days.loc[x.index]].mean() if up_days.loc[x.index].any() else 0, raw=False
    )
    data['down_day_volume_ma'] = data['volume'].rolling(window=5).apply(
        lambda x: x[down_days.loc[x.index]].mean() if down_days.loc[x.index].any() else 0, raw=False
    )
    data['volume_asymmetry'] = (data['up_day_volume_ma'] - data['down_day_volume_ma']) / (
        data['up_day_volume_ma'] + data['down_day_volume_ma'] + 1e-8)
    
    # Volume-weighted price move asymmetry
    data['vw_price_move'] = (data['price_change'] * data['volume']).rolling(window=5).mean()
    data['abs_vw_price_move'] = (data['abs_price_change'] * data['volume']).rolling(window=5).mean()
    data['price_volume_asymmetry'] = data['vw_price_move'] / (data['abs_vw_price_move'] + 1e-8)
    
    # Flow Imbalance Quantification
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['high_close_pressure'] = (data['high'] - data['close']) / (data['true_range'] + 1e-8)
    data['close_low_pressure'] = (data['close'] - data['low']) / (data['true_range'] + 1e-8)
    data['bidirectional_pressure'] = data['close_low_pressure'] - data['high_close_pressure']
    
    # Directional absorption
    data['directional_absorption'] = (data['price_change'] * data['amount']).rolling(window=5).sum()
    
    # 2. Microstructure Efficiency Analysis
    # Range Utilization Assessment
    data['micro_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['range_utilization'] = (data['close'] - data['open']) / (data['true_range'] + 1e-8)
    
    # Efficiency consistency (3-day rolling std of efficiency)
    data['efficiency_consistency'] = 1 / (data['micro_efficiency'].rolling(window=3).std() + 1e-8)
    
    # Value-Density Dislocation
    data['value_density'] = (data['volume'] / (data['amount'] + 1e-8)) * data['close']
    data['value_density_ma'] = data['value_density'].rolling(window=5).mean()
    data['value_density_anomaly'] = data['value_density'] / (data['value_density_ma'] + 1e-8) - 1
    
    # Volume-value divergence
    data['volume_momentum'] = data['volume'].pct_change(periods=3)
    data['price_momentum'] = data['close'].pct_change(periods=3)
    data['volume_value_divergence'] = data['volume_momentum'] - data['price_momentum']
    
    # 3. Asymmetry Acceleration Dynamics
    # Multi-Period Momentum Confluence
    data['asymmetry_momentum_2d'] = data['price_volume_asymmetry'].pct_change(periods=2)
    data['asymmetry_momentum_5d'] = data['price_volume_asymmetry'].pct_change(periods=5)
    data['momentum_resonance'] = np.sign(data['asymmetry_momentum_2d']) * np.sign(data['asymmetry_momentum_5d'])
    
    # Flow Acceleration Patterns
    data['flow_imbalance_momentum'] = data['bidirectional_pressure'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=True
    )
    
    # Volume burst patterns (z-score of volume relative to 10-day window)
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=10).mean()) / (
        data['volume'].rolling(window=10).std() + 1e-8)
    data['volume_burst_response'] = data['volume_zscore'] * data['price_change']
    
    # 4. Microstructural Regime Classification
    # Strong vs Weak Asymmetry Regime
    volume_concentration = data['volume'].rolling(window=5).std() / (data['volume'].rolling(window=5).mean() + 1e-8)
    asymmetry_persistence = data['price_volume_asymmetry'].rolling(window=5).std()
    flow_acceleration = abs(data['flow_imbalance_momentum'])
    
    # Regime score (higher = stronger asymmetry regime)
    data['regime_score'] = (
        (1 / (volume_concentration + 1e-8)) * 0.3 +
        (1 / (asymmetry_persistence + 1e-8)) * 0.4 +
        flow_acceleration * 0.3
    )
    
    # 5. Cross-Association Signal Generation
    # Core Asymmetry Momentum Score
    data['core_asymmetry'] = (
        data['price_volume_asymmetry'] * 0.25 +
        data['bidirectional_pressure'] * 0.20 +
        data['micro_efficiency'] * 0.15 +
        data['value_density_anomaly'] * 0.20 +
        data['flow_imbalance_momentum'] * 0.20
    )
    
    # Regime-conditional signal processing
    strong_regime = data['regime_score'] > data['regime_score'].rolling(window=20).quantile(0.7)
    weak_regime = data['regime_score'] < data['regime_score'].rolling(window=20).quantile(0.3)
    
    # Base momentum with regime adjustments
    data['base_momentum'] = data['core_asymmetry'].rolling(window=5).mean()
    
    # Strong regime: emphasize acceleration and persistence
    strong_signal = data['base_momentum'] * (1 + data['flow_imbalance_momentum'])
    
    # Weak regime: require multi-confirmation
    weak_confirmation = (
        data['momentum_resonance'] +
        np.sign(data['directional_absorption']) +
        np.sign(data['volume_value_divergence'])
    )
    weak_signal = data['base_momentum'] * (weak_confirmation / 3)
    
    # Final factor with regime conditioning
    data['asymmetry_momentum_factor'] = np.where(
        strong_regime, strong_signal,
        np.where(weak_regime, weak_signal, data['base_momentum'])
    )
    
    # Normalize the final factor
    rolling_mean = data['asymmetry_momentum_factor'].rolling(window=20).mean()
    rolling_std = data['asymmetry_momentum_factor'].rolling(window=20).std()
    data['final_factor'] = (data['asymmetry_momentum_factor'] - rolling_mean) / (rolling_std + 1e-8)
    
    return data['final_factor']
