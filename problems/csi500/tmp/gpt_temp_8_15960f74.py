import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor: Multi-Day Trend Strength with Volume-Price Divergence and Adaptive Volatility Scaling
    
    Interpretable components:
    1. Multi-day trend strength: Combines short-term (3-day) and medium-term (10-day) momentum
    2. Volume-price divergence: Detects when price movements are supported by volume trends
    3. Adaptive volatility scaling: Uses rolling volatility relative to recent volatility regime
    4. Trend consistency: Measures how sustained the current trend has been
    
    Factor logic: Stocks showing strong, consistent multi-day trends with volume confirmation
    and stable volatility conditions tend to exhibit return continuation.
    """
    # Multi-day momentum components
    short_term_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    medium_term_momentum = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Volume-price divergence: volume trend relative to price trend
    volume_trend = df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()
    price_trend = df['close'] / df['close'].rolling(window=10, min_periods=5).mean()
    volume_price_divergence = volume_trend / (price_trend + 1e-7)
    
    # Adaptive volatility scaling using rolling standard deviation
    returns = df['close'].pct_change()
    current_volatility = returns.rolling(window=5, min_periods=3).std()
    regime_volatility = returns.rolling(window=20, min_periods=10).std()
    volatility_ratio = regime_volatility / (current_volatility + 1e-7)
    
    # Trend consistency: how many of recent days were in the same direction
    daily_returns = df['close'].pct_change()
    trend_direction = np.sign(short_term_momentum)
    consistent_days = trend_direction.rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    trend_consistency = consistent_days / 5.0
    
    # Composite factor with interpretable interactions
    alpha_factor = (
        (0.6 * short_term_momentum + 0.4 * medium_term_momentum) *  # Multi-day trend blend
        np.tanh(volume_price_divergence) *  # Bounded volume-price relationship
        np.sqrt(volatility_ratio) *  # Square root for moderate volatility scaling
        trend_consistency  # Trend persistence multiplier
    )
    
    return alpha_factor
