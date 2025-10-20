import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Multi-Timeframe Efficiency Momentum
    # 5-day efficiency
    high_5d = data['high'].rolling(window=5, min_periods=5).max()
    low_5d = data['low'].rolling(window=5, min_periods=5).min()
    efficiency_5d = (data['close'] - data['close'].shift(5)) / (high_5d - low_5d + 1e-8)
    
    # 10-day efficiency
    high_10d = data['high'].rolling(window=10, min_periods=10).max()
    low_10d = data['low'].rolling(window=10, min_periods=10).min()
    efficiency_10d = (data['close'] - data['close'].shift(10)) / (high_10d - low_10d + 1e-8)
    
    # Efficiency momentum
    eff_momentum = efficiency_5d - efficiency_10d
    
    # Efficiency divergence (variance across multiple timeframes)
    efficiency_3d = (data['close'] - data['close'].shift(3)) / (data['high'].rolling(3).max() - data['low'].rolling(3).min() + 1e-8)
    efficiency_8d = (data['close'] - data['close'].shift(8)) / (data['high'].rolling(8).max() - data['low'].rolling(8).min() + 1e-8)
    eff_divergence = pd.concat([efficiency_3d, efficiency_5d, efficiency_8d, efficiency_10d], axis=1).var(axis=1)
    
    # Volume-Adapted Range Analysis
    volume_adj_range = (data['high'] - data['low']) * np.log(data['volume'] + 1)
    range_efficiency = (data['close'] - data['close'].shift(1)) / (volume_adj_range + 1e-8)
    
    # Volume persistence (5-day autocorrelation)
    volume_5d = data['volume'].rolling(window=5, min_periods=5)
    volume_persistence = volume_5d.apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # Price-Volume Synchronization
    # 10-day price-volume efficiency correlation
    price_changes = data['close'].pct_change()
    volume_changes = data['volume'].pct_change()
    eff_correlation = price_changes.rolling(window=10, min_periods=10).corr(volume_changes)
    
    # Structural break (5-day/20-day efficiency variance ratio)
    eff_5d_var = efficiency_5d.rolling(window=5, min_periods=5).var()
    eff_20d_var = efficiency_5d.rolling(window=20, min_periods=20).var()
    structural_break = eff_5d_var / (eff_20d_var + 1e-8)
    
    # Synchronization strength
    sync_strength = eff_correlation * np.where(structural_break > 1, 1, 0)
    
    # Volatility-Regime Adaptation
    # Efficiency regime
    eff_20d_avg = efficiency_5d.rolling(window=20, min_periods=20).mean()
    efficiency_regime = np.where(efficiency_5d > eff_20d_avg, 1, -1)
    
    # Volatility assessment
    avg_range_20d = (data['high'] - data['low']).rolling(window=20, min_periods=20).mean()
    volatility_assessment = (data['high'] - data['low']) / (avg_range_20d + 1e-8)
    
    # Adaptive weighting
    vol_regime = np.where(volatility_assessment > 1, 1.2, 0.8)
    eff_regime_weight = np.where(efficiency_regime == 1, 1.5, 0.7)
    
    # Acceleration Patterns
    # Efficiency acceleration (second derivative)
    eff_acceleration = efficiency_5d.diff().diff()
    
    # Volume confirmation
    volume_trend = data['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    volume_confirmation = np.sign(eff_acceleration) * np.sign(volume_trend)
    
    # Break enhancement
    break_enhancement = structural_break * np.abs(eff_acceleration)
    
    # Composite Alpha Construction
    # Base efficiency momentum with volume weighting
    base_signal = eff_momentum * (1 + volume_persistence)
    
    # Volume-adjusted range signals
    range_signal = range_efficiency * volume_adj_range.rolling(window=5, min_periods=5).mean()
    
    # Structural break enhancement
    break_signal = sync_strength * break_enhancement
    
    # Volatility-regime filtering
    regime_filtered = base_signal * vol_regime * eff_regime_weight
    
    # Acceleration confirmation
    accel_confirmation = eff_acceleration * volume_confirmation
    
    # Final composite alpha
    composite_alpha = (
        0.4 * regime_filtered +
        0.25 * range_signal +
        0.2 * break_signal +
        0.15 * accel_confirmation -
        0.1 * eff_divergence
    )
    
    return composite_alpha
