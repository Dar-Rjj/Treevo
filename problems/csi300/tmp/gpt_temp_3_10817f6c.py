import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Volatility Regime Identification
    vol_20d = returns.rolling(window=20).std()
    vol_60d_median = returns.rolling(window=60).std().rolling(window=60).median()
    volatility_regime = (vol_20d > vol_60d_median).astype(int)
    
    # Multi-Period Breakout Analysis
    # Support-Resistance Framework
    high_50d = df['high'].rolling(window=50).max()
    low_50d = df['low'].rolling(window=50).min()
    
    # Calculate distance to nearest key level
    dist_to_high = (high_50d - df['close']) / df['close']
    dist_to_low = (df['close'] - low_50d) / df['close']
    nearest_level_dist = pd.concat([dist_to_high, dist_to_low], axis=1).min(axis=1)
    
    # Volume-Price Confirmation
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    volume_surge = (df['volume'] > 2 * volume_20d_avg).astype(int)
    
    # Price impact during volume surges
    price_change = df['close'].pct_change()
    volume_surge_impact = (volume_surge * price_change).rolling(window=5).mean()
    
    # Breakout Quality Assessment
    # Range Efficiency Component
    daily_range = (df['high'] - df['low']) / df['close']
    closing_efficiency = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    range_efficiency = closing_efficiency * (1 - daily_range)
    
    # Multi-Day Persistence
    volume_3d_avg = df['volume'].rolling(window=3).mean()
    volume_consistency = (df['volume'] > volume_3d_avg).rolling(window=3).sum() / 3
    
    # Price maintains breakout level (above 50-day high or below 50-day low)
    above_high = (df['close'] > high_50d.shift(1)).astype(int)
    below_low = (df['close'] < low_50d.shift(1)).astype(int)
    breakout_persistence = (above_high + below_low).rolling(window=3).sum() / 3
    
    # Momentum-Volume Convergence Detection
    # Multi-Period Price Momentum
    mom_3d = df['close'].pct_change(3)
    mom_10d = df['close'].pct_change(10)
    mom_20d = df['close'].pct_change(20)
    
    # Multi-Period Volume Momentum
    vol_mom_3d = df['volume'].pct_change(3)
    vol_mom_10d = df['volume'].pct_change(10)
    vol_mom_20d = df['volume'].pct_change(20)
    
    # Detect Momentum-Volume Convergence
    short_term_align = (mom_3d * vol_mom_3d > 0).astype(int)
    medium_term_align = (mom_10d * vol_mom_10d > 0).astype(int)
    long_term_align = (mom_20d * vol_mom_20d > 0).astype(int)
    
    # Calculate Momentum-Volume Convergence Strength
    aligned_count = short_term_align + medium_term_align + long_term_align
    momentum_magnitude = (abs(mom_3d) + abs(mom_10d) + abs(mom_20d)) / 3
    convergence_score = aligned_count * momentum_magnitude
    
    # Intraday Structure Integration
    # Session-Based Analysis
    morning_strength = (df['high'] - df['open']) / df['open']
    afternoon_behavior = (df['close'] - df['high']) / df['high']
    
    # Gap Analysis
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_momentum_alignment = (overnight_gap * morning_strength > 0).astype(int)
    
    # Volume Pattern Recognition
    volume_5d_std = df['volume'].rolling(window=5).std()
    volume_clustering = (df['volume'] > df['volume'].rolling(window=5).mean() + volume_5d_std).astype(int)
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Adaptive Signal Generation
    # Breakout Strength
    breakout_strength = (1 - nearest_level_dist) * volume_surge_impact * breakout_persistence
    
    # Volume Expansion Filter
    volume_deviation = (df['volume'] - volume_20d_avg) / volume_20d_avg
    volume_filter = np.tanh(volume_deviation)
    
    # Intraday Momentum Validation
    session_consistency = (morning_strength * afternoon_behavior > 0).astype(int)
    intraday_validation = (session_consistency + gap_momentum_alignment) / 2
    
    # Composite Signal Construction
    base_signal = breakout_strength * convergence_score * range_efficiency
    filtered_signal = base_signal * volume_filter * intraday_validation
    
    # Regime-Based Final Adjustment
    # High volatility: focus on short-term, reduce persistence
    high_vol_signal = filtered_signal * (1 - 0.3 * breakout_persistence)
    # Low volatility: emphasize convergence, enhance persistence
    low_vol_signal = filtered_signal * (1 + 0.2 * convergence_score) * (1 + 0.3 * breakout_persistence)
    
    # Combine based on regime
    final_signal = volatility_regime * high_vol_signal + (1 - volatility_regime) * low_vol_signal
    
    return final_signal
