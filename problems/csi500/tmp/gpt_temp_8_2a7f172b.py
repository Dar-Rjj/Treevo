import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-adaptive multi-factor alpha combining momentum, volume pressure, 
    timeframe convergence, liquidity efficiency, and volatility breakout signals.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # 1. Regime-Adaptive Momentum Factor
    # Raw momentum calculations
    mom_1d = df['close'].pct_change(1)
    mom_3d = df['close'].pct_change(3)
    mom_10d = df['close'].pct_change(10)
    
    # Volatility regime detection
    true_range = np.maximum(df['high'] - df['low'], 
                           np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    avg_range_20d = true_range.rolling(window=20).mean()
    range_ratio = true_range / avg_range_20d
    
    # Volume-volatility correlation (20-day rolling)
    vol_corr = df['volume'].rolling(window=20).corr(true_range)
    
    # Regime classification
    high_vol_regime = ((range_ratio > 1.2) & (vol_corr > 0.3)).fillna(False)
    low_vol_regime = ((range_ratio < 0.8) & (df['volume'] < df['volume'].rolling(20).mean() * 0.8)).fillna(False)
    
    # Regime-adaptive scaling
    momentum_combined = 0.4 * mom_1d + 0.35 * mom_3d + 0.25 * mom_10d
    regime_scaling = pd.Series(1.0, index=df.index)
    regime_scaling[high_vol_regime] = 0.7  # Dampen in high volatility
    regime_scaling[low_vol_regime] = 1.3   # Amplify in low volatility
    
    regime_momentum = momentum_combined * regime_scaling
    
    # 2. Volume Asymmetry Pressure Factor
    # Up/down day identification
    up_days = df['close'] > df['close'].shift(1)
    down_days = df['close'] < df['close'].shift(1)
    
    # Rolling volume calculations
    up_volume_10d = df['volume'].rolling(window=10).apply(
        lambda x: x[up_days.iloc[-10:].values].mean() if up_days.iloc[-10:].sum() > 0 else 0, 
        raw=False
    )
    down_volume_10d = df['volume'].rolling(window=10).apply(
        lambda x: x[down_days.iloc[-10:].values].mean() if down_days.iloc[-10:].sum() > 0 else 0, 
        raw=False
    )
    
    # Volume concentration ratios
    up_volume_ratio = up_volume_10d / df['volume'].rolling(10).mean()
    down_volume_ratio = down_volume_10d / df['volume'].rolling(10).mean()
    
    # Pressure imbalance
    volume_pressure = (up_volume_ratio - down_volume_ratio) * np.sign(mom_3d)
    
    # 3. Multi-Timeframe Convergence Factor
    # Ultra-short horizon (1-2 days)
    price_accel = mom_1d - mom_1d.shift(1)
    gap_behavior = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Short-term horizon (3-7 days)
    trend_3d = df['close'].rolling(3).apply(lambda x: (x[-1] - x[0]) / x[0], raw=True)
    trend_7d = df['close'].rolling(7).apply(lambda x: (x[-1] - x[0]) / x[0], raw=True)
    volume_trend = df['volume'].pct_change(5)
    
    # Medium-term horizon (8-20 days)
    trend_20d = df['close'].pct_change(20)
    vol_persistence = range_ratio.rolling(5).std()  # Lower = more persistent regime
    
    # Convergence scoring
    timeframe_signals = pd.DataFrame({
        'ultra_short': np.sign(price_accel + gap_behavior),
        'short_term': np.sign(trend_3d + trend_7d),
        'medium_term': np.sign(trend_20d)
    }).fillna(0)
    
    convergence_score = timeframe_signals.sum(axis=1) / 3  # -1 to +1 scale
    
    # 4. Liquidity-Efficiency Price Factor
    # Range utilization
    daily_range = df['high'] - df['low']
    close_move = abs(df['close'] - df['close'].shift(1))
    range_utilization = close_move / (daily_range + 1e-8)
    
    # Gap efficiency
    overnight_gap = abs(df['open'] - df['close'].shift(1))
    gap_fill_ratio = (overnight_gap - abs(df['close'] - df['open'])) / (overnight_gap + 1e-8)
    
    # Price path smoothness (lower = smoother)
    price_volatility = df['close'].pct_change().rolling(5).std()
    
    # Volume-efficiency relationship
    volume_efficiency = range_utilization / (df['volume'] / df['volume'].rolling(20).mean() + 1e-8)
    
    liquidity_efficiency = (range_utilization * 0.4 + 
                           (1 - price_volatility) * 0.3 + 
                           volume_efficiency * 0.3)
    
    # 5. Volatility Breakout Anticipation Factor
    # Compression metrics
    range_contraction = (avg_range_20d - true_range) / avg_range_20d
    volume_drying = (df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean() - 1)
    
    # Breakout positioning
    mid_price = (df['high'] + df['low']) / 2
    range_position = (df['close'] - df['low']) / (daily_range + 1e-8) - 0.5  # -0.5 to +0.5
    
    # Breakout probability
    compression_score = (range_contraction * 0.6 + volume_drying * 0.4)
    positioning_score = abs(range_position)  # Higher when near boundaries
    
    breakout_anticipation = compression_score * positioning_score * np.sign(range_position)
    
    # Final alpha combination with regime-aware weighting
    for date in df.index:
        if pd.notna(regime_momentum.loc[date]) and pd.notna(volume_pressure.loc[date]):
            # Dynamic weighting based on volatility regime
            if high_vol_regime.loc[date]:
                weights = [0.25, 0.20, 0.25, 0.15, 0.15]  # Conservative in high vol
            elif low_vol_regime.loc[date]:
                weights = [0.30, 0.25, 0.20, 0.15, 0.10]  # Aggressive in low vol
            else:
                weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # Normal regime
            
            alpha.loc[date] = (
                weights[0] * regime_momentum.loc[date] +
                weights[1] * volume_pressure.loc[date] +
                weights[2] * convergence_score.loc[date] +
                weights[3] * liquidity_efficiency.loc[date] +
                weights[4] * breakout_anticipation.loc[date]
            )
    
    # Normalize final alpha
    alpha = (alpha - alpha.rolling(20).mean()) / (alpha.rolling(20).std() + 1e-8)
    
    return alpha
