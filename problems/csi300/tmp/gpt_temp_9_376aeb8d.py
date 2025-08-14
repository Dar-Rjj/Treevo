import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Momentum - 50 to 100 period return based on the current market trend, using a dynamic window size
    short_momentum = df['close'].pct_change(50).ewm(span=50, adjust=False).mean()
    long_momentum = df['close'].pct_change(100).ewm(span=100, adjust=False).mean()
    momentum = (short_momentum + long_momentum) / 2

    # Liquidity - Calculate the exponentially weighted moving average of volume over a 60 day period
    liquidity = df['volume'].ewm(span=60, adjust=False).mean()

    # Volatility - Calculate the exponentially weighted standard deviation of daily returns over a 30 day period
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.ewm(span=30, adjust=False).std()

    # True Range (TR) calculation for volatility using a dynamic window size
    prev_close = df['close'].shift(1)
    tr1 = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr1 + tr2 + tr3).ewm(span=30, adjust=False).mean()  # Use exponentially weighted mean for TR

    # Money Flow Index (MFI) with a 20-day exponentially weighted period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).ewm(span=20, adjust=False).mean()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).ewm(span=20, adjust=False).mean()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))

    # Market Sentiment - Calculate the ratio of high to low prices, using a 10-day exponentially weighted moving average for smoothing
    sentiment = (df['high'] / df['low']).ewm(span=10, adjust=False).mean()

    # Price-Volume Interaction
    price_volume_interaction = df['close'] * df['volume']

    # Adaptive Window Size for Momentum
    trend = df['close'].ewm(span=20, adjust=False).mean()
    trend_diff = trend.diff().ewm(span=10, adjust=False).mean()
    adaptive_window = 50 + (trend_diff.abs() * 100).round().astype(int).clip(upper=50)
    adaptive_momentum = df['close'].pct_change(periods=adaptive_window).ewm(span=adaptive_window, adjust=False).mean()

    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction * adaptive_momentum
    return alpha_factor
