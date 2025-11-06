import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Enhanced momentum factor with volume extremes and volatility-adjusted efficiency
    # Economic rationale: Stocks with strong momentum accompanied by exceptional volume activity
    # and efficient price discovery in low-volatility environments tend to exhibit persistent trends
    
    # 15-day price momentum for medium-term trend capture
    momentum = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    
    # Volume extreme indicator: current volume percentile over 20-day window
    volume_rank = df['volume'].rolling(window=20).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    
    # Price efficiency: ratio of absolute returns to trading range over 10 days
    daily_efficiency = abs(df['close'] - df['close'].shift(1)) / ((df['high'] - df['low']) + 1e-7)
    efficiency_trend = daily_efficiency.rolling(window=10).mean()
    
    # Volatility adjustment: inverse of 20-day average true range normalized by price
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    volatility_adjustment = 1 / (true_range.rolling(window=20).mean() / df['close'].rolling(window=20).mean() + 1e-7)
    
    # Combine components: momentum amplified by volume extremes and efficiency, adjusted for low volatility
    factor = momentum * volume_rank * efficiency_trend * volatility_adjustment
    
    return factor
