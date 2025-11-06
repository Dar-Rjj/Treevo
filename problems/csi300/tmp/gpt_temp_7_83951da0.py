import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Range-Efficiency Momentum with Pressure-Volume Confirmation
    """
    data = df.copy()
    
    # Volatility Regime Classification
    # Rolling Range Volatility Calculation
    daily_range = (data['high'] - data['low']) / data['close']
    range_vol_20d = daily_range.rolling(window=20).mean()
    range_vol_std = daily_range.rolling(window=20).std()
    
    # Regime Threshold Definition
    vol_percentile_33 = range_vol_20d.rolling(window=60).apply(lambda x: np.percentile(x.dropna(), 33), raw=False)
    vol_percentile_66 = range_vol_20d.rolling(window=60).apply(lambda x: np.percentile(x.dropna(), 66), raw=False)
    
    # Regime Classification
    low_vol_regime = (range_vol_20d <= vol_percentile_33).astype(int)
    high_vol_regime = (range_vol_20d >= vol_percentile_66).astype(int)
    normal_vol_regime = ((range_vol_20d > vol_percentile_33) & (range_vol_20d < vol_percentile_66)).astype(int)
    
    # Regime Transition Detection
    regime_change = (low_vol_regime.diff().abs() + high_vol_regime.diff().abs() + normal_vol_regime.diff().abs())
    recent_regime_change = regime_change.rolling(window=5).sum()
    regime_stable = (recent_regime_change == 0).astype(int)
    regime_transition = (recent_regime_change > 0).astype(int)
    
    # Range Efficiency Acceleration Analysis
    # Multi-Timeframe Range Efficiency
    range_efficiency = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    eff_short = range_efficiency.rolling(window=3).mean()
    eff_medium = range_efficiency.rolling(window=8).mean()
    eff_long = range_efficiency.rolling(window=20).mean()
    
    # Efficiency Momentum Calculation
    eff_mom_short = eff_short - eff_medium
    eff_mom_medium = eff_medium - eff_long
    eff_acceleration = eff_mom_short / (eff_medium.abs().replace(0, np.nan))
    
    # Efficiency Divergence Patterns
    eff_divergence = eff_mom_short - eff_mom_medium
    
    # Intraday Pressure Momentum Analysis
    # Multi-Timeframe Pressure Calculation
    buying_pressure = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    selling_pressure = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    net_pressure = buying_pressure - selling_pressure
    
    # Pressure Momentum Divergence
    pressure_short = net_pressure.diff(5)
    pressure_long = net_pressure.diff(20)
    pressure_divergence = pressure_short - pressure_long
    
    # Pressure-Efficiency Alignment
    pressure_efficiency_alignment = pressure_divergence * eff_acceleration
    
    # Volume Acceleration Confirmation
    # Volume Efficiency Assessment
    volume_efficiency = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    vol_eff_short = volume_efficiency.rolling(window=3).mean()
    vol_eff_long = volume_efficiency.rolling(window=20).mean()
    
    # Volume Momentum Patterns
    vol_momentum = vol_eff_short - vol_eff_long
    vol_eff_medium = volume_efficiency.rolling(window=8).mean()
    vol_acceleration = (vol_eff_short - vol_eff_medium) / (vol_eff_medium.abs().replace(0, np.nan))
    vol_pressure_alignment = vol_momentum * pressure_divergence
    
    # Core Signal Generation
    raw_efficiency_signal = eff_acceleration * pressure_divergence
    volume_confirmation = raw_efficiency_signal * vol_momentum
    combined_momentum = eff_acceleration * pressure_divergence * vol_acceleration
    
    # Regime-Adaptive Weighting
    # Low Volatility Regime Logic
    low_vol_signal = (
        eff_divergence * 0.6 +  # Focus on efficiency mean-reversion
        pressure_divergence * 0.8 +  # Emphasize pressure divergence
        vol_acceleration * 0.4  # Volume as breakout confirmation
    )
    
    # High Volatility Regime Logic
    high_vol_signal = (
        eff_acceleration * 0.8 +  # Prioritize trend persistence
        pressure_divergence * 0.4 +  # Pressure as momentum filter
        combined_momentum * 0.6  # Risk-adjusted combined signal
    )
    
    # Normal Volatility Regime Logic
    normal_vol_signal = (
        raw_efficiency_signal * 0.7 +
        volume_confirmation * 0.5 +
        combined_momentum * 0.3
    )
    
    # Transition Period Logic
    transition_signal = (
        pressure_efficiency_alignment * 0.4 +
        vol_pressure_alignment * 0.3 +
        eff_acceleration * 0.3
    )
    
    # Volatility Context Integration
    volatility_scaling = 1 / (range_vol_20d.replace(0, np.nan))
    
    # Final Alpha Generation with Regime-Adaptive Blending
    regime_adaptive_signal = (
        low_vol_regime * low_vol_signal * 0.8 +
        high_vol_regime * high_vol_signal * 0.6 +
        normal_vol_regime * normal_vol_signal * 1.0 +
        regime_transition * transition_signal * 0.5
    )
    
    # Apply volatility scaling and regime stability filtering
    final_alpha = (
        regime_adaptive_signal * 
        volatility_scaling * 
        (regime_stable * 1.0 + regime_transition * 0.7)
    )
    
    # Smooth the final signal
    final_alpha_smoothed = final_alpha.rolling(window=3).mean()
    
    return final_alpha_smoothed
