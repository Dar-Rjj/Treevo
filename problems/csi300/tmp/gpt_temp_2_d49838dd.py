import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Percentile-based momentum acceleration with volume divergence and multiplicative smoothing.
    
    Interpretation:
    - Momentum acceleration measured through percentile ranking of price changes
    - Volume divergence detects abnormal trading activity relative to recent patterns
    - Multiplicative smoothing across timeframes enhances signal stability
    - Volatility-scaled volume conviction adjusts for market conditions
    - Regime persistence ensures signal consistency across market states
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest bearish momentum acceleration with volume distribution
    """
    
    # Price range normalization for momentum signals
    price_range = df['high'] - df['low']
    
    # Percentile-based momentum components
    intraday_return = (df['close'] - df['open']) / price_range
    daily_return = (df['close'] - df['close'].shift(1)) / price_range
    
    # Momentum acceleration using rolling percentiles
    intraday_momentum = intraday_return.rolling(window=5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    daily_momentum = daily_return.rolling(window=5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Momentum acceleration factor
    momentum_accel = (intraday_momentum + daily_momentum) * (intraday_momentum - daily_momentum.rolling(window=3).mean())
    
    # Volume divergence detection
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_divergence = (volume_ma_5 / volume_ma_20) * (df['volume'] / volume_ma_5)
    
    # Volatility-scaled volume conviction
    range_volatility = price_range.rolling(window=10).std()
    volume_conviction = volume_divergence / (range_volatility + 1e-7)
    
    # Multiplicative smoothing across timeframes
    short_term_smooth = momentum_accel.rolling(window=3).apply(lambda x: x.prod())
    medium_term_smooth = momentum_accel.rolling(window=5).apply(lambda x: x.prod())
    
    # Regime persistence detection
    momentum_persistence = (momentum_accel > momentum_accel.rolling(window=5).mean()).astype(int)
    volume_persistence = (volume_conviction > volume_conviction.rolling(window=5).mean()).astype(int)
    regime_strength = momentum_persistence + volume_persistence
    
    # Combined alpha factor with multiplicative blending
    alpha_factor = (
        short_term_smooth * 
        medium_term_smooth * 
        volume_conviction * 
        (1 + 0.2 * regime_strength)
    )
    
    return alpha_factor
