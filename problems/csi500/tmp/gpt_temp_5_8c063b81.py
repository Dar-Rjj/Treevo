import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining multiple market microstructure signals
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # 1. Short-Term Volatility-Efficient Momentum
    # 3-Day Sharpe Momentum
    returns_3d = data['close'].pct_change(periods=3)
    vol_3d = data['close'].pct_change().rolling(window=3).std()
    sharpe_momentum = returns_3d / (vol_3d + 1e-8)
    
    # 5-Day Acceleration Ratio
    momentum_2d = data['close'].pct_change(periods=2)
    momentum_5d = data['close'].pct_change(periods=5)
    acceleration_ratio = momentum_2d / (momentum_5d + 1e-8)
    
    # Intraweek Momentum Persistence
    daily_returns = data['close'].pct_change()
    momentum_persistence = pd.Series(0.0, index=data.index)
    for i in range(2, len(data)):
        recent_returns = daily_returns.iloc[i-2:i+1]
        same_sign = ((recent_returns > 0).all() | (recent_returns < 0).all())
        if same_sign:
            count = 3
            avg_magnitude = recent_returns.abs().mean()
            momentum_persistence.iloc[i] = count * avg_magnitude
    
    # 2. Volume-Price Regime Efficiency
    # High-Volume Momentum Confirmation
    vol_median_5d = data['volume'].rolling(window=5).median()
    high_vol_mask = data['volume'] > vol_median_5d
    high_vol_returns = data['close'].pct_change() * high_vol_mask
    high_vol_momentum = high_vol_returns.rolling(window=3).sum()
    
    # Low-Volume Reversal Detection
    low_vol_mask = data['volume'] < vol_median_5d
    low_vol_reversals = -data['close'].pct_change() * low_vol_mask
    low_vol_reversal_sum = low_vol_reversals.rolling(window=2).sum()
    
    # Volume-Price Divergence Efficiency
    price_momentum_3d = data['close'].pct_change(periods=3)
    volume_momentum_3d = data['volume'].pct_change(periods=3)
    volume_divergence = price_momentum_3d / (volume_momentum_3d + 1e-8)
    
    # 3. Intraday Pattern Strength
    # Gap-Fill Momentum Efficiency
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_fill_ratio = (data['close'] - data['open']) / (overnight_gap + 1e-8)
    gap_fill_momentum = overnight_gap * gap_fill_ratio
    
    # Morning-Afternoon Trend Consistency
    morning_strength = (data['high'] - data['open']) / (data['open'] + 1e-8)
    afternoon_follow = (data['close'] - data['low']) / (data['low'] + 1e-8)
    trend_consistency = morning_strength * afternoon_follow
    
    # Intraday Range Efficiency
    range_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    efficiency_momentum = range_efficiency / range_efficiency.rolling(window=3).mean()
    intraday_efficiency = range_efficiency * efficiency_momentum
    
    # 4. Volatility-Regime Adaptive Signals
    # Range Expansion Momentum
    daily_range = (data['high'] - data['low']) / data['close']
    range_avg_5d = daily_range.rolling(window=5).mean()
    range_expansion = daily_range > range_avg_5d
    range_momentum = daily_range * range_expansion * data['close'].pct_change()
    
    # Volatility-Adaptive Breakout
    vol_5d = data['close'].pct_change().rolling(window=5).std()
    vol_median_10d = vol_5d.rolling(window=10).median()
    high_vol_regime = vol_5d > vol_median_10d
    vol_breakout = data['close'].pct_change() * high_vol_regime
    
    # Support/Resistance Efficiency
    resistance_7d = data['high'].rolling(window=7).max().shift(1)
    support_7d = data['low'].rolling(window=7).min().shift(1)
    resistance_break = (data['close'] - resistance_7d) / (resistance_7d + 1e-8) * data['volume']
    support_bounce = (data['close'] - support_7d) / (support_7d + 1e-8) * data['amount']
    
    # 5. Multi-Timeframe Volume Confirmation
    # Volume Direction Alignment
    price_direction = np.sign(data['close'] - data['close'].shift(1))
    volume_direction = np.sign(data['volume'] - data['volume'].shift(1))
    volume_alignment = price_direction * volume_direction * data['close'].pct_change().abs()
    
    # Volume-Weighted Short-Term Momentum
    momentum_2d = data['close'].pct_change(periods=2)
    volume_weighted_return = momentum_2d * data['volume']
    volume_weighted_momentum = volume_weighted_return.rolling(window=3).sum()
    
    # Volume Regime Persistence
    volume_persistence = pd.Series(0.0, index=data.index)
    high_vol_streak = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if data['volume'].iloc[i] > vol_median_5d.iloc[i]:
            high_vol_streak.iloc[i] = high_vol_streak.iloc[i-1] + 1
        else:
            high_vol_streak.iloc[i] = 0
    
    # Calculate average return during high-volume streaks
    for i in range(len(data)):
        if high_vol_streak.iloc[i] > 0:
            streak_start = i - high_vol_streak.iloc[i] + 1
            streak_returns = daily_returns.iloc[streak_start:i+1]
            avg_return = streak_returns.mean()
            volume_persistence.iloc[i] = high_vol_streak.iloc[i] * avg_return
    
    # Combine all factors with equal weights
    factors = [
        sharpe_momentum, acceleration_ratio, momentum_persistence,
        high_vol_momentum, low_vol_reversal_sum, volume_divergence,
        gap_fill_momentum, trend_consistency, intraday_efficiency,
        range_momentum, vol_breakout, resistance_break, support_bounce,
        volume_alignment, volume_weighted_momentum, volume_persistence
    ]
    
    # Z-score normalization and combination
    normalized_factors = []
    for f in factors:
        if f.notna().any():
            z_score = (f - f.mean()) / (f.std() + 1e-8)
            normalized_factors.append(z_score)
    
    # Equal-weighted combination
    if normalized_factors:
        factor = sum(normalized_factors) / len(normalized_factors)
    
    return factor
