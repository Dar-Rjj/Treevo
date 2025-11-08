import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Weighted Trend-Range Divergence factor
    """
    data = df.copy()
    
    # Calculate Multi-Timeframe Trend Components
    # Short-Term Trend Persistence
    close_up = (data['close'] > data['close'].shift(1)).astype(int)
    close_down = (data['close'] < data['close'].shift(1)).astype(int)
    
    # 3-day directional streak components
    up_streak_3d = close_up.rolling(window=3, min_periods=1).sum()
    down_streak_3d = close_down.rolling(window=3, min_periods=1).sum()
    trend_persistence = up_streak_3d - down_streak_3d
    
    # Medium-Term Trend Acceleration
    momentum_3d = data['close'] / data['close'].shift(3) - 1
    momentum_8d = data['close'] / data['close'].shift(8) - 1
    trend_acceleration = momentum_3d - momentum_8d
    
    # Trend Consistency Score
    trend_consistency = trend_persistence * trend_acceleration
    
    # Calculate Range-Volume Divergence Components
    # Daily True Range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Range Expansion Ratio
    avg_tr_3d = true_range.rolling(window=3, min_periods=1).mean()
    avg_tr_8d = true_range.rolling(window=8, min_periods=1).mean()
    range_expansion_ratio = avg_tr_3d / avg_tr_8d
    
    # Volume-Range Efficiency
    abs_return = abs(data['close'] / data['close'].shift(1) - 1)
    volume_range_efficiency = (abs_return / true_range) * data['volume']
    
    # Range-Volume Divergence Score
    range_volume_divergence = range_expansion_ratio * volume_range_efficiency
    
    # Calculate Volatility-Weighted Gap Behavior
    # Daily Gap Magnitude
    gap_magnitude = abs(data['open'] / data['close'].shift(1) - 1)
    
    # Gap direction (1 for gap up, -1 for gap down, 0 for no gap)
    gap_direction = np.sign(data['open'] - data['close'].shift(1))
    gap_direction = gap_direction.replace(0, np.nan)
    
    # Gap persistence (consecutive same-sign gaps)
    gap_persistence = gap_direction.groupby(
        (gap_direction != gap_direction.shift(1)).cumsum()
    ).cumcount() + 1
    
    # Average gap magnitude over persistence period
    gap_magnitude_avg = gap_magnitude.rolling(window=3, min_periods=1).mean()
    
    # Volatility adjustment (10-day ATR)
    atr_10d = true_range.rolling(window=10, min_periods=1).mean()
    
    # Volatility-Adjusted Gap Persistence
    gap_persistence_vol_adj = (gap_persistence * gap_magnitude_avg) / atr_10d
    
    # Gap-Range Alignment
    daily_range_movement = np.sign(data['close'] - data['open'])
    gap_range_alignment = (gap_direction == daily_range_movement).astype(float)
    gap_range_alignment = gap_range_alignment.replace(0, -1)  # -1 for misalignment
    
    gap_behavior = gap_persistence_vol_adj * gap_range_alignment * gap_magnitude
    
    # Volatility Regime Detection
    atr_percentile = atr_10d.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 66)) * 2 + (x.iloc[-1] > np.percentile(x, 33)) * 1,
        raw=False
    )
    
    # Regime-Adaptive Component Combination
    high_vol_weight = (atr_percentile == 2).astype(float)
    low_vol_weight = (atr_percentile == 0).astype(float)
    medium_vol_weight = (atr_percentile == 1).astype(float)
    
    # Volatility-Weighted Signal Processing
    # Inverse volatility scaling for trend components
    trend_scaled = trend_consistency / (atr_10d + 1e-8)
    
    # Direct volatility scaling for range-based components
    range_volume_scaled = range_volume_divergence * atr_10d
    gap_behavior_scaled = gap_behavior * atr_10d
    
    # Final Composite Factor with regime-adaptive weighting
    factor = (
        (trend_scaled * (low_vol_weight + 0.5 * medium_vol_weight)) +
        (range_volume_scaled * (high_vol_weight + 0.5 * medium_vol_weight)) +
        (gap_behavior_scaled * (high_vol_weight + medium_vol_weight))
    )
    
    return factor
