import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor: Volatility-Normalized Multi-Timeframe Momentum-Volume Composite with Directional Filtering
    
    This factor combines momentum and volume signals across short and medium timeframes (3-day and 7-day),
    applies directional consistency filtering, then normalizes by volatility to create a risk-adjusted signal.
    
    Interpretation:
    - Positive values: Strong bearish momentum with volume confirmation and directional consistency
    - Negative values: Strong bullish momentum with volume confirmation and directional consistency
    - Magnitude reflects signal strength relative to market volatility
    - Inversion provides contrarian characteristics for portfolio diversification
    
    Economic rationale:
    - Multi-timeframe alignment confirms trend consistency across different horizons
    - Volume ratios validate institutional participation and conviction
    - Directional filtering reduces noise from inconsistent price movements
    - Volatility normalization adjusts for different market regimes and risk levels
    - Simple construction maintains interpretability while capturing complex market dynamics
    """
    
    # Calculate momentum across two timeframes (3-day and 7-day)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_7d = (df['close'] - df['close'].shift(7)) / df['close'].shift(7)
    
    # Calculate volume ratios for corresponding timeframes
    volume_ratio_3d = df['volume'] / df['volume'].rolling(window=3, min_periods=2).mean()
    volume_ratio_7d = df['volume'] / df['volume'].rolling(window=7, min_periods=4).mean()
    
    # Directional consistency filter (at least 2 out of 3 recent days in same direction)
    daily_returns = df['close'].pct_change()
    recent_direction = daily_returns.rolling(window=3, min_periods=2).apply(
        lambda x: 1 if (x > 0).sum() >= 2 else (-1 if (x < 0).sum() >= 2 else 0)
    )
    
    # Calculate return volatility using 7-day standard deviation
    return_volatility = daily_returns.rolling(window=7, min_periods=4).std()
    
    # Combine momentum and volume components multiplicatively for each timeframe
    momentum_volume_3d = momentum_3d * volume_ratio_3d
    momentum_volume_7d = momentum_7d * volume_ratio_7d
    
    # Blend timeframes with equal weights and apply directional filter
    blended_factor = (momentum_volume_3d + momentum_volume_7d) * recent_direction
    
    # Normalize by volatility and invert for diversification
    alpha_factor = -blended_factor / (return_volatility + 1e-7)
    
    return alpha_factor
