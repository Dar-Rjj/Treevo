import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the 5-day Exponential Moving Average (EMA) of the close price
    ema_close = df['close'].ewm(span=5, adjust=False).mean()
    
    # Calculate the 10-day Exponential Moving Average (EMA) of the volume
    ema_volume = df['volume'].ewm(span=10, adjust=False).mean()
    
    # Calculate the 5-day rolling standard deviation of the close price as a measure of volatility
    volatility = df['close'].rolling(window=5).std().fillna(0)
    
    # Calculate the momentum as the difference between today's close and the 5-day EMA close
    momentum = (df['close'] - ema_close).fillna(0)
    
    # Compute the relative range of today's trading compared to the last 10 days
    relative_range = (df['high'] - df['low']) / df['close'].shift(1)
    relative_range_rank = relative_range.rank(pct=True)
    
    # Calculate the relative strength as the ratio of the current close to the 10-day minimum close
    relative_strength = df['close'] / df['close'].rolling(window=10).min()
    
    # Calculate the weighted average of the amount over the last 5 days
    weighted_amount = df['amount'].rolling(window=5).apply(lambda x: (x * [1, 2, 3, 4, 5]).sum() / 15).fillna(0)
    
    # Combine the factors: momentum, relative range rank, and relative strength, adjusted by volatility and weighted amount
    factor = (momentum * relative_range_rank * relative_strength) / (volatility + 1e-6) * (ema_volume / weighted_amount)
    
    return factor
