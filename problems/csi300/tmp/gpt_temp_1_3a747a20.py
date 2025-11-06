import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Price Structure Analysis
    # Intraday Price Efficiency
    daily_efficiency = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    efficiency_persistence = daily_efficiency.rolling(window=3, min_periods=1).mean()
    
    # Multi-Period Price Compression
    high_5d = data['high'].rolling(window=5, min_periods=1).max()
    low_5d = data['low'].rolling(window=5, min_periods=1).min()
    high_20d = data['high'].rolling(window=20, min_periods=1).max()
    low_20d = data['low'].rolling(window=20, min_periods=1).min()
    price_range_ratio = (high_5d - low_5d) / (high_20d - low_20d).replace(0, np.nan)
    
    high_10d = data['high'].rolling(window=10, min_periods=1).max()
    low_10d = data['low'].rolling(window=10, min_periods=1).min()
    current_10d_range = high_10d - low_10d
    prev_10d_range = current_10d_range.shift(1)
    range_acceleration = (current_10d_range - prev_10d_range) / prev_10d_range.replace(0, np.nan)
    range_acceleration = range_acceleration * np.sign(data['close'] / data['close'].shift(10) - 1)
    
    # Price Structure Composite
    price_compression = price_range_ratio * range_acceleration
    price_structure_composite = np.cbrt(efficiency_persistence * price_compression)
    
    # Volume Flow Asymmetry Detection
    # Bidirectional Volume Pressure
    mid_price = (data['open'] + data['close']) / 2
    upward_pressure = np.where(data['close'] > mid_price, data['volume'], 0)
    downward_pressure = np.where(data['close'] < mid_price, data['volume'], 0)
    
    total_daily_volume = data['volume'].rolling(window=5, min_periods=1).mean()
    upward_pressure_norm = upward_pressure / total_daily_volume.replace(0, np.nan)
    downward_pressure_norm = downward_pressure / total_daily_volume.replace(0, np.nan)
    
    # Volume Flow Imbalance
    net_volume_flow = upward_pressure_norm - downward_pressure_norm
    net_flow_5d_avg = net_volume_flow.rolling(window=5, min_periods=1).mean()
    net_flow_5d_std = net_volume_flow.rolling(window=5, min_periods=1).std()
    volume_flow_momentum = (net_volume_flow - net_flow_5d_avg) / net_flow_5d_std.replace(0, np.nan)
    
    # Volume Asymmetry Signal
    volume_asymmetry_signal = np.tanh(volume_flow_momentum * price_structure_composite)
    
    # Multi-Scale Momentum Divergence
    # Short-Range Momentum Structure
    momentum_3d_accel = (data['close'] / data['close'].shift(3) - 
                        data['close'].shift(3) / data['close'].shift(6))
    price_range_3d = data['high'].rolling(window=3, min_periods=1).max() - data['low'].rolling(window=3, min_periods=1).min()
    momentum_3d_accel = momentum_3d_accel * price_range_3d
    
    momentum_5d_curvature = (data['close'] / data['close'].shift(5) - 
                           2 * data['close'].shift(5) / data['close'].shift(10) + 
                           data['close'].shift(10) / data['close'].shift(15))
    atr_5d = (data['high'].rolling(window=5, min_periods=1).max() - 
             data['low'].rolling(window=5, min_periods=1).min()).rolling(window=5, min_periods=1).mean()
    momentum_5d_curvature = momentum_5d_curvature / atr_5d.replace(0, np.nan)
    
    # Medium-Range Momentum Structure
    returns_2d = data['close'].pct_change(periods=2)
    sign_consistency = returns_2d.rolling(window=10, min_periods=1).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) > 0 else np.nan
    )
    
    momentum_20d_efficiency = (data['close'] / data['close'].shift(20) - 1).abs()
    daily_returns_abs = data['close'].pct_change().abs()
    sum_abs_returns = daily_returns_abs.rolling(window=20, min_periods=1).sum()
    momentum_20d_efficiency = momentum_20d_efficiency / sum_abs_returns.replace(0, np.nan)
    momentum_20d_efficiency = momentum_20d_efficiency * np.sign(data['close'] / data['close'].shift(20) - 1)
    
    # Momentum Structure Composite
    momentum_component1 = momentum_3d_accel * sign_consistency
    momentum_component2 = momentum_5d_curvature * momentum_20d_efficiency
    momentum_structure_composite = (momentum_component1 + momentum_component2) / 2
    
    # Opening Auction Anomaly Detection
    opening_gap = data['open'] - data['close'].shift(1)
    prev_day_range = data['high'].shift(1) - data['low'].shift(1)
    gap_efficiency = opening_gap / prev_day_range.replace(0, np.nan)
    opening_volume_ratio = data['volume'] / data['volume'].shift(1).replace(0, np.nan)
    gap_efficiency = gap_efficiency * opening_volume_ratio
    
    # First-hour price absorption (simplified using daily high/low)
    first_hour_absorption = np.where(
        opening_gap > 0,
        (data['high'] - data['open']) / opening_gap.abs().replace(0, np.nan),
        (data['open'] - data['low']) / opening_gap.abs().replace(0, np.nan)
    )
    
    auction_quality = gap_efficiency * first_hour_absorption
    auction_quality = 1 / (1 + np.exp(-auction_quality))
    
    # Auction-Adjusted Factor
    auction_adjusted = volume_asymmetry_signal * auction_quality + momentum_structure_composite
    
    # Regime-Independent Alpha Core
    # Price-Volume Alignment Score
    price_changes = data['close'].pct_change()
    volume_flow_changes = net_volume_flow.diff()
    
    def sign_correlation(x, y):
        if len(x) < 2:
            return np.nan
        return np.corrcoef(np.sign(x), np.sign(y))[0, 1]
    
    price_volume_alignment = price_changes.rolling(window=10, min_periods=1).apply(
        lambda x: sign_correlation(x, volume_flow_changes.loc[x.index])
    )
    
    # Momentum-Volume Convergence
    momentum_normalized = momentum_structure_composite / momentum_structure_composite.rolling(window=20, min_periods=1).std().replace(0, np.nan)
    volume_normalized = volume_asymmetry_signal / volume_asymmetry_signal.rolling(window=20, min_periods=1).std().replace(0, np.nan)
    momentum_volume_convergence = np.sqrt(momentum_normalized * volume_normalized)
    
    # Asymmetry Amplification Engine
    signal_values = auction_adjusted.rolling(window=20, min_periods=1)
    positive_ratio = signal_values.apply(lambda x: np.sum(x > 0) / len(x) if len(x) > 0 else np.nan)
    asymmetry_persistence = positive_ratio.rolling(window=5, min_periods=1).std()
    
    def asymmetric_weight(x, positive_ratio_val):
        if positive_ratio_val > 0.5:
            return x * (1 + positive_ratio_val)
        else:
            return x * (1 - positive_ratio_val)
    
    asymmetry_weighted = auction_adjusted.rolling(window=1).apply(
        lambda x: asymmetric_weight(x.iloc[0], positive_ratio.loc[x.index[0]]) if len(x) > 0 else np.nan
    )
    
    # Core Alpha Factor
    integrated_signals = (price_volume_alignment + momentum_volume_convergence) / 2
    core_alpha = integrated_signals * asymmetry_weighted
    core_alpha = np.arctan(core_alpha)
    
    # Dynamic Signal Refinement
    # Volatility-Adaptive Smoothing
    vol_5d = data['close'].pct_change().rolling(window=5, min_periods=1).std()
    vol_20d = data['close'].pct_change().rolling(window=20, min_periods=1).std()
    vol_10d = data['close'].pct_change().rolling(window=10, min_periods=1).std()
    vol_60d = data['close'].pct_change().rolling(window=60, min_periods=1).std()
    
    vol_ratio_5_20 = vol_5d / vol_20d.replace(0, np.nan)
    vol_ratio_10_60 = vol_10d / vol_60d.replace(0, np.nan)
    
    def adaptive_smoothing(x, vol_ratio_5_20_val, vol_ratio_10_60_val):
        avg_vol_ratio = (vol_ratio_5_20_val + vol_ratio_10_60_val) / 2
        if avg_vol_ratio > 1.2:
            return x.rolling(window=3, min_periods=1).mean().iloc[-1]
        elif avg_vol_ratio < 0.8:
            return x.rolling(window=10, min_periods=1).mean().iloc[-1]
        else:
            return x.rolling(window=5, min_periods=1).mean().iloc[-1]
    
    smoothed_alpha = core_alpha.rolling(window=1).apply(
        lambda x: adaptive_smoothing(core_alpha.loc[:x.index[0]], 
                                   vol_ratio_5_20.loc[x.index[0]], 
                                   vol_ratio_10_60.loc[x.index[0]]) if len(x) > 0 else np.nan
    )
    
    # Cross-sectional ranking enhancement
    def cross_sectional_rank(x):
        if len(x) < 5:
            return np.nan
        return (x.rank(pct=True) - 0.5).iloc[-1]
    
    cross_sectional_signal = smoothed_alpha.rolling(window=20, min_periods=1).apply(cross_sectional_rank)
    
    # Final Multi-Frequency Alpha
    final_alpha = smoothed_alpha * (1 + cross_sectional_signal)
    
    return final_alpha
