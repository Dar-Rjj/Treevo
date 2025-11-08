import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining volatility-adjusted momentum, 
    volume-price convergence, liquidity breakthroughs, filtered mean reversion,
    and intraday persistence patterns.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Adjusted Momentum Breakout
    # Compute 5-day momentum
    momentum_5d = data['close'].pct_change(periods=5)
    
    # Calculate 20-day Average True Range (ATR) for volatility
    high_low = data['high'] - data['low']
    high_close_prev = np.abs(data['high'] - data['close'].shift(1))
    low_close_prev = np.abs(data['low'] - data['close'].shift(1))
    true_range = np.maximum(np.maximum(high_low, high_close_prev), low_close_prev)
    atr_20d = true_range.rolling(window=20, min_periods=1).mean()
    
    # Adjust momentum by volatility regime (inverse relationship)
    volatility_adjusted_momentum = momentum_5d / (atr_20d + 1e-8)
    
    # Volume-Price Acceleration Convergence
    # Volume acceleration (current vs previous day ratio)
    volume_acceleration = data['volume'] / (data['volume'].shift(1) + 1e-8)
    
    # Price trend slope over 3 days using linear regression
    def price_slope_3d(window):
        if len(window) < 3:
            return 0
        x = np.arange(len(window))
        return np.polyfit(x, window, 1)[0] / window.iloc[0]
    
    price_trend_slope = data['close'].rolling(window=3, min_periods=1).apply(
        price_slope_3d, raw=False
    )
    
    # Volume-price convergence factor
    volume_price_convergence = volume_acceleration * price_trend_slope
    
    # Liquidity Breakthrough Strength
    # Recent support/resistance levels (10-day high/low)
    resistance_10d = data['high'].rolling(window=10, min_periods=1).max()
    support_10d = data['low'].rolling(window=10, min_periods=1).min()
    
    # Detect breakthroughs and calculate strength
    breakthrough_up = (data['close'] > resistance_10d.shift(1)).astype(int)
    breakthrough_down = (data['close'] < support_10d.shift(1)).astype(int)
    
    # Breakthrough distance (normalized by price)
    breakthrough_distance_up = (data['close'] - resistance_10d.shift(1)) / data['close']
    breakthrough_distance_down = (support_10d.shift(1) - data['close']) / data['close']
    
    # Volume surge detection (>150% of 20-day average)
    volume_20d_avg = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_surge = (data['volume'] > 1.5 * volume_20d_avg).astype(int)
    
    # Combined breakthrough strength
    breakthrough_strength = (
        breakthrough_up * breakthrough_distance_up * volume_surge -
        breakthrough_down * breakthrough_distance_down * volume_surge
    )
    
    # Filtered Mean Reversion
    # Price deviation from 20-day moving average
    ma_20d = data['close'].rolling(window=20, min_periods=1).mean()
    price_deviation = (data['close'] - ma_20d) / ma_20d
    
    # 5-day average high-low range for volatility
    range_5d_avg = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    range_normalized = range_5d_avg / data['close']
    
    # Adjust deviation by volatility (stronger in high volatility)
    filtered_mean_reversion = price_deviation * range_normalized
    
    # Intraday Persistence Pattern
    # Daily range utilization (close position within high-low range)
    daily_range = data['high'] - data['low']
    close_position = (data['close'] - data['low']) / (daily_range + 1e-8)
    
    # 3-day consistency with magnitude weighting
    def persistence_score(window):
        if len(window) < 3:
            return 0
        # Weight by magnitude of range utilization deviation from 0.5
        weights = np.abs(window - 0.5)
        return np.corrcoef(np.arange(len(window)), window)[0, 1] * np.mean(weights)
    
    range_persistence = close_position.rolling(window=3, min_periods=1).apply(
        persistence_score, raw=False
    )
    
    # Combine all factors with equal weighting
    alpha_factor = (
        0.2 * volatility_adjusted_momentum +
        0.2 * volume_price_convergence +
        0.2 * breakthrough_strength +
        0.2 * filtered_mean_reversion +
        0.2 * range_persistence
    )
    
    # Normalize the final factor
    alpha_factor = (alpha_factor - alpha_factor.mean()) / (alpha_factor.std() + 1e-8)
    
    return alpha_factor
