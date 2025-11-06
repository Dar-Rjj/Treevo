import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-horizon momentum-volume factor with volatility scaling and decay weighting.
    Combines short (1-day), medium (5-day), and long-term (20-day) momentum signals,
    weighted by volume trends and scaled by volatility, with exponential decay.
    """
    # Multi-horizon momentum components
    mom_short = df['close'] / df['close'].shift(1) - 1
    mom_medium = df['close'] / df['close'].shift(5) - 1
    mom_long = df['close'] / df['close'].shift(20) - 1
    
    # Volume trend (5-day moving average normalized by 20-day average)
    volume_trend = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    
    # Volatility scaling (20-day rolling standard deviation of returns)
    volatility = mom_short.rolling(20).std()
    
    # Range efficiency penalty (measures inefficiency in price movement)
    daily_range = (df['high'] - df['low']) / df['close']
    range_efficiency = abs(mom_short) / (daily_range + 1e-7)
    inefficiency_penalty = 1.0 - range_efficiency
    
    # Exponential decay weights (recent observations get higher weight)
    decay_weights = np.exp(-np.arange(20) / 5.0)  # 5-day decay half-life
    decay_weights = decay_weights / decay_weights.sum()
    
    # Apply decay weights to momentum components
    mom_short_weighted = mom_short.rolling(20).apply(lambda x: np.sum(x * decay_weights))
    mom_medium_weighted = mom_medium.rolling(20).apply(lambda x: np.sum(x * decay_weights))
    mom_long_weighted = mom_long.rolling(20).apply(lambda x: np.sum(x * decay_weights))
    
    # Combine components with volume trend amplification and inefficiency penalty
    alpha_factor = (
        (0.4 * mom_short_weighted + 0.35 * mom_medium_weighted + 0.25 * mom_long_weighted) 
        * volume_trend 
        * inefficiency_penalty 
        / (volatility + 1e-7)
    )
    
    return alpha_factor
