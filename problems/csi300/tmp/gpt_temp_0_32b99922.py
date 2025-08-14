import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Momentum - 50 to 100 period return based on the current market trend
    short_momentum = df['close'].pct_change(50)
    long_momentum = df['close'].pct_change(100)
    momentum = (short_momentum + long_momentum) / 2

    # Liquidity - Calculate the exponentially weighted moving average of volume over a 60 day period
    liquidity = df['volume'].ewm(span=60).mean()

    # Volatility - Calculate the exponentially weighted standard deviation of daily returns over a 30 day period
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.ewm(span=30).std()

    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs().combine((df['high'] - prev_close).abs(), max).combine((prev_close - df['low']).abs(), max)
    avg_true_range = tr.ewm(span=30).mean()  # Use exponentially weighted mean for TR

    # Money Flow Index (MFI) with a 20-day exponentially weighted period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).ewm(span=20).mean()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).ewm(span=20).mean()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))

    # Market Sentiment - Calculate the ratio of high to low prices, using a 10-day exponentially weighted moving average for smoothing
    sentiment = (df['high'] / df['low']).ewm(span=10).mean()

    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1)
    return alpha_factor
