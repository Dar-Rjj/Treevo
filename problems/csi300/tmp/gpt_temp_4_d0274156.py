import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Momentum - Dynamic window size based on the 50 to 100 period return
    short_momentum = df['close'].pct_change(50)
    long_momentum = df['close'].pct_change(100)
    momentum = (short_momentum + long_momentum) / 2

    # Liquidity - Exponential moving average of volume over a 60 day period
    liquidity = df['volume'].ewm(span=60, adjust=False).mean()

    # Volatility - Rolling standard deviation of daily returns with exponential weighting
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.ewm(span=30, adjust=False).std()

    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).ewm(span=30, adjust=False).mean()  # Use EWM for TR

    # Money Flow Index (MFI) with a 20-day exponential weighted moving average
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).ewm(span=20, adjust=False).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).ewm(span=20, adjust=False).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))

    # Market Sentiment - Ratio of high to low prices, using a 10-day EMA for smoothing
    sentiment = (df['high'] / df['low']).ewm(span=10, adjust=False).mean()

    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1)
    return alpha_factor
