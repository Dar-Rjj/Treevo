import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Horizon Range Efficiency Analysis
    daily_range_ratio = (df['close'] - df['open']) / (df['high'] - df['low'])
    daily_range_ratio = daily_range_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    efficiency_3d = daily_range_ratio.rolling(window=3, min_periods=1).mean()
    efficiency_8d = daily_range_ratio.rolling(window=8, min_periods=1).mean()
    efficiency_divergence = efficiency_3d - efficiency_8d
    
    # Volume-Price Asymmetry Detection
    momentum_3d = df['close'].pct_change(periods=3)
    
    # Calculate Up-Day vs Down-Day Volume Ratio
    returns = df['close'].pct_change()
    up_days = returns > 0
    down_days = returns < 0
    
    up_volume_10d = df['volume'].rolling(window=10).apply(
        lambda x: x[up_days.loc[x.index]].sum() if up_days.loc[x.index].any() else 0, raw=False
    )
    down_volume_10d = df['volume'].rolling(window=10).apply(
        lambda x: x[down_days.loc[x.index]].sum() if down_days.loc[x.index].any() else 0, raw=False
    )
    
    volume_asymmetry_ratio = up_volume_10d / (down_volume_10d + 1e-8)
    volume_asymmetry_ratio = volume_asymmetry_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Identify divergence between momentum and volume asymmetry
    momentum_volume_divergence = momentum_3d * (volume_asymmetry_ratio - volume_asymmetry_ratio.rolling(window=10, min_periods=1).mean())
    
    # Liquidity-Weighted Breakout Quality
    # Identify 15-day range breakout patterns
    high_15d = df['high'].rolling(window=15, min_periods=1).max()
    low_15d = df['low'].rolling(window=15, min_periods=1).min()
    
    breakout_up = df['close'] > high_15d.shift(1)
    breakout_down = df['close'] < low_15d.shift(1)
    
    # Calculate Volume×Amount liquidity score
    liquidity_score = df['volume'] * df['amount']
    avg_liquidity_10d = liquidity_score.rolling(window=10, min_periods=1).mean()
    
    breakout_liquidity_ratio = liquidity_score / (avg_liquidity_10d + 1e-8)
    
    # Breakout quality considering direction
    breakout_quality = np.where(breakout_up, breakout_liquidity_ratio, 
                               np.where(breakout_down, -breakout_liquidity_ratio, 0))
    
    # Intraday Gap Efficiency Integration
    gap_magnitude = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    
    # Measure intraday momentum following gap
    intraday_momentum = (df['close'] - df['open']) / (df['open'] + 1e-8)
    gap_momentum_efficiency = gap_magnitude * intraday_momentum
    
    # Track gap fill speed relative to volume conditions
    gap_fill_speed = np.abs(gap_magnitude) / (df['volume'] + 1e-8)
    gap_fill_speed_normalized = gap_fill_speed / (gap_fill_speed.rolling(window=10, min_periods=1).mean() + 1e-8)
    
    # Adaptive Composite Alpha Generation
    # Multiply multi-horizon efficiency by volume asymmetry magnitude
    base_signal = efficiency_divergence * np.abs(momentum_volume_divergence)
    
    # Weight by breakout liquidity confirmation
    weighted_signal = base_signal * breakout_quality
    
    # Scale by gap-driven intraday momentum
    scaled_signal = weighted_signal * gap_momentum_efficiency
    
    # Validate signal persistence over 3-day window
    final_alpha = scaled_signal.rolling(window=3, min_periods=1).mean()
    
    return final_alpha
