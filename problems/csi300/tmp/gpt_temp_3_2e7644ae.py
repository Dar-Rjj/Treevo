import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Percentile-based momentum acceleration with volume divergence and multiplicative smoothing.
    
    Interpretation:
    - Momentum acceleration measured through percentile ranking across multiple timeframes
    - Volume divergence identifies unusual trading activity relative to recent patterns
    - Multiplicative smoothing combines short and medium-term signals for persistence
    - Volatility-scaled volume conviction adjusts for market conditions
    - Price range normalization ensures comparability across different price levels
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest deteriorating momentum with distribution patterns
    """
    
    # Price range normalized momentum components
    price_range = df['high'] - df['low'] + 1e-7
    intraday_momentum = (df['close'] - df['open']) / price_range
    daily_momentum = (df['close'] - df['close'].shift(1)) / price_range
    
    # Percentile-based momentum acceleration (5-day vs 20-day)
    momentum_5d = daily_momentum.rolling(window=5).mean()
    momentum_20d = daily_momentum.rolling(window=20).mean()
    
    # Calculate rolling percentiles for acceleration measurement
    momentum_accel = momentum_5d.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.5)) / (x.quantile(0.75) - x.quantile(0.25) + 1e-7)
    )
    
    # Volume divergence: current volume vs recent distribution
    volume_divergence = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.5)) / (x.quantile(0.75) - x.quantile(0.25) + 1e-7)
    )
    
    # Volatility-scaled volume conviction
    daily_volatility = price_range.rolling(window=5).std()
    volume_conviction = volume_divergence / (daily_volatility + 1e-7)
    
    # Multiplicative smoothing across timeframes
    short_term_signal = momentum_accel * np.sign(intraday_momentum)
    medium_term_signal = momentum_5d.rolling(window=5).mean() * np.sign(momentum_5d)
    
    # Regime persistence through multiplicative combination
    alpha_factor = (
        short_term_signal * medium_term_signal * 
        np.sign(volume_conviction) * (1 + 0.1 * np.abs(volume_conviction))
    )
    
    return alpha_factor
