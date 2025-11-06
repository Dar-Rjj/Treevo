import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Percentile-based momentum acceleration with volume divergence and multiplicative smoothing.
    
    Interpretation:
    - Momentum acceleration measured through percentile ranking of price changes
    - Volume divergence detects unusual trading activity relative to recent patterns
    - Multiplicative smoothing across different timeframes enhances signal stability
    - Volatility scaling adapts to current market conditions without normalization
    - Price range normalization provides consistent momentum measurement
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest accelerating bearish pressure with volume divergence
    """
    
    # Price range for momentum normalization
    price_range = df['high'] - df['low'] + 1e-7
    
    # Momentum components with range normalization
    intraday_return = (df['close'] - df['open']) / price_range
    overnight_return = (df['open'] - df['close'].shift(1)) / price_range
    daily_return = (df['close'] - df['close'].shift(1)) / price_range
    
    # Momentum acceleration using percentile ranking
    intraday_accel = intraday_return.rolling(window=5).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 1.0 + (x.iloc[-1] < x.quantile(0.3)) * -1.0, raw=False)
    daily_accel = daily_return.rolling(window=5).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 1.0 + (x.iloc[-1] < x.quantile(0.3)) * -1.0, raw=False)
    
    # Volume divergence using percentile comparison
    volume_rank = df['volume'].rolling(window=10).apply(lambda x: (x.iloc[-1] > x.quantile(0.8)) * 1.0 + (x.iloc[-1] < x.quantile(0.2)) * -1.0, raw=False)
    amount_rank = df['amount'].rolling(window=10).apply(lambda x: (x.iloc[-1] > x.quantile(0.8)) * 1.0 + (x.iloc[-1] < x.quantile(0.2)) * -1.0, raw=False)
    
    # Volume divergence signal
    volume_divergence = volume_rank * amount_rank * np.sign(volume_rank * amount_rank)
    
    # Multiplicative smoothing across timeframes
    short_term_momentum = (intraday_return + overnight_return) * np.sign(intraday_return * overnight_return)
    medium_term_momentum = daily_return.rolling(window=3).mean()
    
    # Volatility scaling using rolling range
    volatility_scale = price_range.rolling(window=5).std() / (price_range.rolling(window=20).mean() + 1e-7)
    
    # Combined momentum with acceleration hierarchy
    momentum_acceleration = (
        intraday_accel * short_term_momentum * 0.4 +
        daily_accel * medium_term_momentum * 0.3 +
        (intraday_accel + daily_accel) * np.sign(intraday_accel * daily_accel) * 0.3
    )
    
    # Final alpha factor with volume divergence and volatility scaling
    alpha_factor = (
        momentum_acceleration * 0.7 +
        volume_divergence * 0.3 * np.sign(momentum_acceleration)
    ) * volatility_scale
    
    return alpha_factor
