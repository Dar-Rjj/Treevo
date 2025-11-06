import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Divergence Alpha Factor
    """
    data = df.copy()
    
    # Volatility Regime Detection
    # True Range Calculation
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Rolling Percentile Analysis (20-day window)
    tr_10pct = true_range.rolling(window=20, min_periods=10).quantile(0.1)
    tr_90pct = true_range.rolling(window=20, min_periods=10).quantile(0.9)
    
    # Regime Classification
    low_vol_regime = (true_range < tr_10pct).astype(int)
    high_vol_regime = (true_range > tr_90pct).astype(int)
    normal_vol_regime = ((true_range >= tr_10pct) & (true_range <= tr_90pct)).astype(int)
    
    # Multi-Timeframe Momentum Analysis
    # Short-Term Momentum Components
    high_2d_momentum = data['high'] / data['high'].shift(2) - 1
    low_5d_momentum = data['low'] / data['low'].shift(5) - 1
    short_term_spread = high_2d_momentum - low_5d_momentum
    
    # Medium-Term Momentum Components
    return_10d = data['close'] / data['close'].shift(10) - 1
    return_20d = data['close'] / data['close'].shift(20) - 1
    medium_term_acceleration = return_10d - return_20d
    
    # Volume Efficiency Dynamics
    # Daily Volume Efficiency
    volume_efficiency = data['amount'] / data['volume']
    volume_efficiency = volume_efficiency.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    # Volume Persistence (3-day slope)
    volume_3d_slope = data['volume'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / 2 if len(x) == 3 else np.nan
    )
    
    # Volume Acceleration Components
    volume_5d_sum = data['volume'].rolling(window=5).sum()
    volume_10d_sum = data['volume'].rolling(window=10).sum()
    volume_acceleration = volume_5d_sum - volume_10d_sum
    
    # Volume Trend Divergence
    volume_3d_slope_abs = data['volume'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / 2 if len(x) == 3 else np.nan
    )
    volume_10d_slope = data['volume'].rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / 9 if len(x) == 10 else np.nan
    )
    volume_trend_divergence = (
        np.sign(volume_3d_slope_abs) * np.sign(volume_10d_slope) * 
        abs(volume_3d_slope_abs - volume_10d_slope)
    )
    
    # Range Efficiency Divergence
    range_efficiency = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    range_efficiency = range_efficiency.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    range_efficiency_mean = range_efficiency.rolling(window=20, min_periods=10).mean()
    range_efficiency_divergence = range_efficiency - range_efficiency_mean
    
    # Divergence Pattern Recognition
    # Momentum thresholds
    strong_momentum = (short_term_spread > short_term_spread.rolling(window=20).quantile(0.7))
    weak_momentum = (short_term_spread < short_term_spread.rolling(window=20).quantile(0.3))
    
    # Volume efficiency thresholds
    high_volume_efficiency = (volume_efficiency > volume_efficiency.rolling(window=20).quantile(0.7))
    low_volume_efficiency = (volume_efficiency < volume_efficiency.rolling(window=20).quantile(0.3))
    
    # Pattern classification
    pattern_a = (strong_momentum & high_volume_efficiency).astype(int)
    pattern_b = (strong_momentum & low_volume_efficiency).astype(int)
    pattern_c = (weak_momentum & high_volume_efficiency).astype(int)
    pattern_d = (weak_momentum & low_volume_efficiency).astype(int)
    
    # Core Divergence Factor
    base_component = short_term_spread * volume_efficiency
    acceleration_component = medium_term_acceleration * volume_acceleration
    core_divergence_factor = 0.6 * base_component + 0.4 * acceleration_component
    
    # Regime-Weighted Component Integration
    low_vol_weight = 1.2
    normal_weight = 1.0
    high_vol_weight = 0.8
    
    regime_weight = (
        low_vol_regime * low_vol_weight + 
        normal_vol_regime * normal_weight + 
        high_vol_regime * high_vol_weight
    )
    
    # Pattern-Based Multiplier Application
    pattern_multiplier = (
        pattern_a * 1.2 +
        pattern_b * 0.8 +
        pattern_c * -1.0 +
        pattern_d * 0.5
    )
    
    # Ensure at least one pattern is active
    pattern_mask = (pattern_a + pattern_b + pattern_c + pattern_d) > 0
    pattern_multiplier = pattern_multiplier.where(pattern_mask, 1.0)
    
    # Final Alpha Generation
    base_signal = core_divergence_factor * regime_weight
    final_alpha = base_signal * pattern_multiplier
    
    # Add volume trend and range efficiency divergences as enhancements
    enhancement_factor = 0.3 * volume_trend_divergence + 0.2 * range_efficiency_divergence
    final_alpha = final_alpha + enhancement_factor
    
    # Normalize and clean
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    final_alpha = final_alpha.fillna(method='ffill')
    
    # Z-score normalization over 60-day window
    alpha_mean = final_alpha.rolling(window=60, min_periods=20).mean()
    alpha_std = final_alpha.rolling(window=60, min_periods=20).std()
    final_alpha = (final_alpha - alpha_mean) / alpha_std
    
    return final_alpha
