import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Percentile-based momentum acceleration with volume divergence and multiplicative smoothing.
    
    Interpretation:
    - Momentum acceleration measured through percentile ranks across multiple timeframes
    - Volume divergence detects abnormal trading activity relative to recent patterns
    - Multiplicative smoothing enhances signal persistence across different market regimes
    - Price range normalization ensures momentum signals are scale-invariant
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest deteriorating momentum with volume distribution patterns
    """
    
    # Price range normalization
    price_range = df['high'] - df['low']
    
    # Momentum components normalized by price range
    intraday_return = (df['close'] - df['open']) / (price_range + 1e-7)
    overnight_return = (df['open'] - df['close'].shift(1)) / (price_range + 1e-7)
    daily_return = (df['close'] - df['close'].shift(1)) / (price_range + 1e-7)
    
    # Percentile-based momentum acceleration
    intraday_momentum = intraday_return.rolling(window=5).apply(lambda x: (x.iloc[-1] > x.quantile(0.6)) * 1.0 + (x.iloc[-1] < x.quantile(0.4)) * -1.0, raw=False)
    overnight_momentum = overnight_return.rolling(window=5).apply(lambda x: (x.iloc[-1] > x.quantile(0.6)) * 1.0 + (x.iloc[-1] < x.quantile(0.4)) * -1.0, raw=False)
    daily_momentum = daily_return.rolling(window=5).apply(lambda x: (x.iloc[-1] > x.quantile(0.6)) * 1.0 + (x.iloc[-1] < x.quantile(0.4)) * -1.0, raw=False)
    
    # Momentum acceleration hierarchy
    ultra_short_accel = (intraday_momentum + overnight_momentum) * (intraday_momentum * overnight_momentum > 0)
    short_term_accel = (daily_momentum + ultra_short_accel) * (daily_momentum * ultra_short_accel > 0)
    
    # Volume divergence detection
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_divergence = (df['volume'] / (volume_ma_5 + 1e-7) - 1.0) * (df['volume'] / (volume_ma_10 + 1e-7) - 1.0)
    
    # Volume-pressure regimes based on divergence persistence
    volume_pressure = volume_divergence.rolling(window=3).apply(lambda x: (x > 0).sum() / 3.0, raw=False)
    
    # Multiplicative smoothing across timeframes
    momentum_smoothed = (intraday_momentum.rolling(window=3).mean() * 
                        overnight_momentum.rolling(window=3).mean() * 
                        daily_momentum.rolling(window=3).mean())
    
    # Regime persistence detection
    momentum_persistence = (intraday_momentum > 0).rolling(window=5).sum() / 5.0
    volume_persistence = (volume_divergence > 0).rolling(window=5).sum() / 5.0
    
    # Volatility-scaled volume conviction
    range_volatility = price_range.rolling(window=5).std()
    volume_conviction = volume_pressure * (1.0 / (range_volatility + 1e-7))
    
    # Core alpha factor construction
    alpha_factor = (
        momentum_smoothed * 0.4 +
        short_term_accel * 0.3 +
        volume_conviction * 0.2 +
        (momentum_persistence * volume_persistence) * 0.1
    )
    
    return alpha_factor
