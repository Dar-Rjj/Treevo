import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators
    """
    # Price-Volume Divergence Momentum
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    open_price = df['open']
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # 1. Price-Volume Divergence Momentum
    price_momentum_5 = close.pct_change(5)
    volume_momentum_5 = volume.pct_change(5)
    
    # Calculate divergence strength
    divergence_strength = np.abs(price_momentum_5 - volume_momentum_5)
    divergence_signal = np.sign(price_momentum_5) * np.sign(volume_momentum_5)
    pv_divergence = price_momentum_5 * (1 - divergence_signal) * divergence_strength
    
    # 2. High-Low Range Efficiency
    true_range = np.maximum(high - low, 
                           np.maximum(np.abs(high - close.shift(1)), 
                                     np.abs(low - close.shift(1))))
    avg_true_range_10 = true_range.rolling(window=10).mean()
    price_movement_5 = np.abs(close.pct_change(5))
    range_efficiency = price_movement_5 / (avg_true_range_10 / close)
    
    # 3. Volume-Sensitive Price Reversal
    price_change = close.pct_change()
    volume_ratio = volume / volume.rolling(window=20).mean()
    
    # Identify overreaction days
    large_move = np.abs(price_change) > price_change.rolling(window=20).std() * 1.5
    high_volume = volume_ratio > 1.5
    
    # Mean reversion signal weighted by volume
    reversal_3 = -close.pct_change(3)
    volume_sensitive_reversal = reversal_3 * volume_ratio * (large_move & high_volume)
    
    # 4. Amount-Per-Volume Momentum
    amount_per_share = amount / volume
    aps_momentum_5 = amount_per_share.pct_change(5)
    aps_price_divergence = aps_momentum_5 - price_momentum_5
    
    # Volume stability weight
    volume_stability = 1 / (volume.rolling(window=10).std() / volume.rolling(window=10).mean())
    aps_signal = aps_price_divergence * volume_stability
    
    # 5. Open-Close Gap Persistence
    daily_gap = (open_price - close.shift(1)) / close.shift(1)
    gap_direction_consistency = daily_gap.rolling(window=5).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 5 else np.nan
    )
    volume_trend = volume.rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    gap_persistence = daily_gap * gap_direction_consistency * np.sign(volume_trend)
    
    # 6. Volatility-Adjusted Volume Flow
    price_direction = np.where(close > close.shift(1), 1, -1)
    volume_flow = volume * price_direction
    price_volatility = (high - low).rolling(window=10).std()
    volatility_adjusted_flow = volume_flow / (price_volatility + 1e-8)
    smoothed_flow = volatility_adjusted_flow.rolling(window=5).mean()
    
    # 7. Multi-Timeframe Price Consistency
    ma_fast = close.rolling(window=5).mean()
    ma_medium = close.rolling(window=10).mean()
    ma_slow = close.rolling(window=20).mean()
    
    trend_alignment = ((ma_fast > ma_medium).astype(int) + 
                      (ma_medium > ma_slow).astype(int) + 
                      (close > ma_fast).astype(int))
    trend_strength = (ma_fast.pct_change(5) + ma_medium.pct_change(10) + ma_slow.pct_change(20)) / 3
    consistency_score = trend_alignment * trend_strength
    
    # 8. Volume-Weighted Support Resistance
    resistance_level = high.rolling(window=20).max()
    support_level = low.rolling(window=20).min()
    
    # Volume concentration near levels
    near_resistance = (close >= resistance_level * 0.98) & (close <= resistance_level * 1.02)
    near_support = (close >= support_level * 0.98) & (close <= support_level * 1.02)
    
    volume_density_resistance = volume.rolling(window=5).mean() * near_resistance
    volume_density_support = volume.rolling(window=5).mean() * near_support
    
    breakout_signal = ((close > resistance_level).astype(int) * volume_density_resistance - 
                      (close < support_level).astype(int) * volume_density_support)
    
    # 9. Amount-Based Order Flow Imbalance
    up_days = close > close.shift(1)
    down_days = close < close.shift(1)
    
    buy_amount = amount * up_days
    sell_amount = amount * down_days
    
    buy_pressure = buy_amount.rolling(window=5).sum()
    sell_pressure = sell_amount.rolling(window=5).sum()
    
    order_flow_ratio = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-8)
    smoothed_flow_ratio = order_flow_ratio.rolling(window=3).mean()
    
    # 10. Range Expansion Contraction Cycle
    range_expansion_rate = true_range.pct_change(5)
    expansion_periods = range_expansion_rate > range_expansion_rate.rolling(window=20).mean()
    contraction_periods = range_expansion_rate < range_expansion_rate.rolling(window=20).mean()
    
    phase_transition = (expansion_periods.astype(int) - contraction_periods.astype(int)).diff()
    range_cycle_signal = phase_transition * volume_ratio
    
    # Combine all signals with weights
    alpha = (0.15 * pv_divergence.rank(pct=True) +
             0.12 * range_efficiency.rank(pct=True) +
             0.13 * volume_sensitive_reversal.rank(pct=True) +
             0.10 * aps_signal.rank(pct=True) +
             0.08 * gap_persistence.rank(pct=True) +
             0.11 * smoothed_flow.rank(pct=True) +
             0.09 * consistency_score.rank(pct=True) +
             0.08 * breakout_signal.rank(pct=True) +
             0.07 * smoothed_flow_ratio.rank(pct=True) +
             0.07 * range_cycle_signal.rank(pct=True))
    
    return alpha
