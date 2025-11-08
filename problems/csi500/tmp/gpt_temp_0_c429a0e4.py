import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor: Lagged Volatility-Adjusted Returns with Volume Trend Weighting
    
    This factor captures:
    1. Lagged volatility-adjusted returns (risk-normalized momentum with time alignment)
    2. Recent volume trend changes (volume momentum as conviction signal)
    3. Price range efficiency (returns relative to daily trading range)
    4. Directional consistency (persistence of price movement)
    
    Interpretation:
    - Uses lagged returns to avoid lookahead bias while maintaining signal relevance
    - Combines volatility normalization with volume trend weighting for robust signals
    - Higher values indicate sustained, low-risk momentum with increasing participation
    - Filters noise by requiring consistent directional movement and efficient price action
    """
    
    # Lagged volatility-adjusted returns (5-day lagged 10-day return scaled by 15-day volatility)
    # Aligns momentum and volatility timeframes with appropriate lag
    lagged_return = df['close'].shift(5) / df['close'].shift(15) - 1
    volatility = df['close'].pct_change().rolling(15).std()
    volatility_adjusted_return = lagged_return / (volatility + 1e-7)
    
    # Volume trend weighting (recent volume changes relative to medium-term average)
    # Captures conviction changes in trading activity
    volume_trend = df['volume'].rolling(5).mean() / df['volume'].rolling(15).mean()
    
    # Price range efficiency (absolute returns relative to average daily range)
    # Measures how efficiently price moves within its typical trading range
    avg_daily_range = (df['high'] - df['low']).rolling(10).mean()
    range_efficiency = lagged_return.abs() / (avg_daily_range / df['close'].shift(5) + 1e-7)
    
    # Directional consistency (persistence of price movement direction)
    # Positive when recent price changes show consistent directional bias
    directional_returns = df['close'].pct_change(periods=3).rolling(5)
    positive_consistency = (directional_returns.apply(lambda x: (x > 0).sum()) / 5.0)
    negative_consistency = (directional_returns.apply(lambda x: (x < 0).sum()) / 5.0)
    directional_consistency = positive_consistency - negative_consistency
    
    # Combine components: lagged volatility-adjusted returns weighted by volume trends
    # Enhanced by range efficiency and directional consistency filters
    alpha_factor = (
        volatility_adjusted_return * 
        volume_trend * 
        range_efficiency * 
        directional_consistency
    )
    
    return alpha_factor
