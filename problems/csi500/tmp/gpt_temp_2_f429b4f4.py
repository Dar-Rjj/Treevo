import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Directional Volume-Price Efficiency
    # Compute Asymmetric Volume Impact
    price_change = data['close'].diff()
    up_days = price_change > 0
    down_days = price_change < 0
    
    # Calculate up-day volume efficiency
    up_volume_sum = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: x[up_days.loc[x.index]].sum() if up_days.loc[x.index].any() else 0
    )
    up_price_change_sum = price_change.rolling(window=5, min_periods=3).apply(
        lambda x: x[up_days.loc[x.index]].sum() if up_days.loc[x.index].any() else 1
    )
    up_volume_efficiency = up_volume_sum / (up_price_change_sum.abs() + 1e-8)
    
    # Calculate down-day volume efficiency
    down_volume_sum = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: x[down_days.loc[x.index]].sum() if down_days.loc[x.index].any() else 0
    )
    down_price_change_sum = price_change.rolling(window=5, min_periods=3).apply(
        lambda x: x[down_days.loc[x.index]].sum() if down_days.loc[x.index].any() else 1
    )
    down_volume_efficiency = down_volume_sum / (down_price_change_sum.abs() + 1e-8)
    
    # Compute volume efficiency asymmetry ratio
    volume_efficiency_asymmetry = (up_volume_efficiency - down_volume_efficiency) / (up_volume_efficiency + down_volume_efficiency + 1e-8)
    
    # Measure Multi-Period Efficiency Persistence
    efficiency_5d = volume_efficiency_asymmetry.rolling(window=5, min_periods=3).mean()
    efficiency_10d = volume_efficiency_asymmetry.rolling(window=10, min_periods=5).mean()
    efficiency_trend = efficiency_5d.diff(3)
    
    # Detect Efficiency Regime Shifts
    efficiency_momentum = efficiency_5d - efficiency_10d
    efficiency_acceleration = efficiency_trend.diff()
    
    # Construct Price-Range Compression-Expansion Cycles
    # Analyze Daily Price Range Dynamics
    true_range = data['high'] - data['low']
    range_5d_avg = true_range.rolling(window=5, min_periods=3).mean()
    compression_ratio = true_range / (range_5d_avg + 1e-8)
    
    # Calculate range expansion momentum
    range_roc = true_range.pct_change(periods=3)
    expansion_persistence = (true_range > true_range.shift(1)).rolling(window=5, min_periods=3).sum()
    
    # Detect Cycle Phase Transitions
    compression_threshold = 0.8
    expansion_threshold = 1.2
    is_compressed = compression_ratio < compression_threshold
    is_expanding = compression_ratio > expansion_threshold
    is_neutral = ~is_compressed & ~is_expanding
    
    # Calculate regime persistence
    regime_persistence = pd.Series(0, index=data.index)
    regime_persistence[is_compressed] = 1
    regime_persistence[is_expanding] = 2
    regime_persistence = regime_persistence.rolling(window=5, min_periods=3).apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 1)
    
    # Implement Amount-Based Order Flow Imbalance
    # Calculate Directional Amount Pressure
    amount_change = data['amount'].diff()
    net_amount_flow = amount_change.rolling(window=3, min_periods=2).sum()
    cumulative_net_amount = net_amount_flow.rolling(window=5, min_periods=3).sum()
    
    # Measure amount volatility
    amount_volatility = data['amount'].rolling(window=5, min_periods=3).std()
    amount_intensity = amount_change.abs() / (amount_volatility + 1e-8)
    
    # Assess Order Flow Persistence
    amount_direction = (amount_change > 0).astype(int)
    consecutive_amount_days = amount_direction.rolling(window=3, min_periods=2).apply(
        lambda x: len(x) if len(set(x)) == 1 else 0
    )
    amount_momentum = data['amount'].pct_change(periods=3)
    
    # Generate Amount-Regime Conditional Signals
    compressed_amount_signal = cumulative_net_amount * is_compressed
    expanding_amount_signal = amount_momentum * is_expanding
    neutral_amount_signal = amount_intensity * is_neutral
    
    # Integrate Multi-Dimensional Asymmetry Patterns
    # Combine Volume Efficiency and Range Dynamics
    joint_asymmetry = volume_efficiency_asymmetry * compression_ratio
    
    # Apply regime-dependent weighting
    regime_weighted_asymmetry = joint_asymmetry.copy()
    regime_weighted_asymmetry[is_compressed] *= 1.5  # Emphasize efficiency in compressed regimes
    regime_weighted_asymmetry[is_expanding] *= 0.8   # Reduce weight in expanding regimes
    regime_weighted_asymmetry[is_neutral] *= 1.2     # Moderate emphasis in neutral
    
    # Incorporate Order Flow Confirmation
    flow_alignment = np.sign(regime_weighted_asymmetry) * np.sign(cumulative_net_amount)
    flow_persisted = consecutive_amount_days > 2
    
    # Apply flow persistence scaling
    flow_scaled_signal = regime_weighted_asymmetry * (1 + 0.2 * flow_persisted)
    flow_confirmed_signal = flow_scaled_signal * (1 + 0.3 * (flow_alignment > 0))
    
    # Generate Regime-Adaptive Composite
    # Create regime-specific signal combinations
    composite_signal = pd.Series(0.0, index=data.index)
    
    # Compressed regime: emphasize efficiency signals
    compressed_mask = regime_persistence == 1
    composite_signal[compressed_mask] = (
        efficiency_momentum[compressed_mask] * 0.6 +
        flow_confirmed_signal[compressed_mask] * 0.4
    )
    
    # Expanding regime: focus on range momentum
    expanding_mask = regime_persistence == 2
    composite_signal[expanding_mask] = (
        range_roc[expanding_mask] * 0.5 +
        expansion_persistence[expanding_mask] * 0.3 +
        flow_confirmed_signal[expanding_mask] * 0.2
    )
    
    # Neutral/transition regimes: weight order flow heavily
    neutral_mask = regime_persistence == 0
    composite_signal[neutral_mask] = (
        cumulative_net_amount[neutral_mask] * 0.5 +
        amount_intensity[neutral_mask] * 0.3 +
        efficiency_trend[neutral_mask] * 0.2
    )
    
    # Calculate dynamic signal strength
    signal_strength = composite_signal.rolling(window=5, min_periods=3).std()
    normalized_composite = composite_signal / (signal_strength + 1e-8)
    
    # Final alpha factor
    alpha_factor = normalized_composite.fillna(0)
    
    return alpha_factor
