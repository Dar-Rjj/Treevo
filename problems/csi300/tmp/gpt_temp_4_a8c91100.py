import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Bounded momentum using relative trends (5-day log returns)
    log_returns = np.log(df['close'] / df['close'].shift(5))
    
    # Calculate rolling momentum percentiles for bounding
    momentum_rank = log_returns.rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    bounded_momentum = np.tanh(momentum_rank * 4 - 2)  # Scale to [-1,1]
    
    # Volume divergence using relative trends
    volume_returns = np.log(df['volume'] / df['volume'].shift(5))
    price_volume_divergence = log_returns - volume_returns
    
    # Normalize divergence using rolling z-score
    divergence_zscore = price_volume_divergence.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-7), raw=False
    )
    volume_signal = np.tanh(divergence_zscore)
    
    # Volatility-scaled efficiency
    daily_range = df['high'] - df['low']
    intraday_efficiency = (df['close'] - df['open']) / (daily_range + 1e-7)
    
    # Volatility regime scaling (high/low volatility periods)
    volatility = daily_range.rolling(window=5).std()
    vol_regime = volatility.rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Scale efficiency by volatility regime (more weight in normal vol periods)
    vol_scaling = 1 + np.cos(vol_regime * np.pi)  # U-shaped weighting
    scaled_efficiency = intraday_efficiency * vol_scaling
    
    # Range position with trend context
    range_position = (df['close'] - df['low']) / (daily_range + 1e-7)
    range_trend = range_position.rolling(window=5).mean()
    range_signal = np.tanh((range_trend - 0.5) * 4)  # Center at 0.5, bound to [-1,1]
    
    # Multiplicative combination with bounded components
    alpha_factor = (
        (1 + bounded_momentum) * 
        (1 + volume_signal) * 
        (1 + scaled_efficiency) * 
        (1 + range_signal)
    ) - 1  # Center around 0
    
    return alpha_factor
