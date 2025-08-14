import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the 2-day Exponential Moving Average (EMA) of the close price for smoothing
    ema_close = df['close'].ewm(span=2, adjust=False).mean().fillna(0)
    
    # Calculate the difference between today's EMA and yesterday's EMA, then scale by the 2-day average of the amount
    momentum_scaled_ema = (ema_close.diff() / df['amount'].rolling(window=2).mean()).fillna(0)
    
    # Compute the relative range of today's trading compared to the last 3 days
    relative_range = (df['high'] - df['low']) / df['close'].shift(1)
    relative_range_rank = relative_range.rank(pct=True).fillna(0)
    
    # Calculate the relative strength as the ratio of the current EMA to the 3-day minimum EMA
    relative_strength = ema_close / ema_close.rolling(window=3).min().fillna(0)
    
    # Calculate the 2-day rolling standard deviation of the EMA close price as a measure of volatility
    volatility_ema = ema_close.rolling(window=2).std().fillna(0)
    
    # Adaptive lookback period for volume-weighted metrics
    adaptive_lookback = df['volume'].rolling(window=5).mean().fillna(0)
    
    # Volume-weighted recent momentum
    volume_weighted_momentum = (df['close'] - df['close'].shift(adaptive_lookback)) / adaptive_lookback
    
    # Combine the factors: momentum scaled EMA, relative range rank, relative strength, adjusted by EMA volatility, and volume-weighted momentum
    factor = (momentum_scaled_ema * relative_range_rank * relative_strength * volume_weighted_momentum) / (volatility_ema + 1e-6)
    
    return factor
