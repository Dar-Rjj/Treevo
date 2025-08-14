import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum calculation - Adaptive window based on volatility
    daily_returns = df['close'].pct_change()
    volatility_30 = daily_returns.rolling(window=30).std()
    adaptive_window = (volatility_30 * 100).round().astype(int)
    momentum = df['close'].pct_change(adaptive_window)

    # Liquidity - Calculate the average volume over an adaptive window
    liquidity = df['volume'].rolling(window=adaptive_window).mean()

    # Volatility - Calculate the rolling standard deviation of daily returns over an adaptive window
    volatility = daily_returns.rolling(window=adaptive_window).std()

    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).rolling(window=adaptive_window).mean()  # Use rolling mean for TR

    # Money Flow Index (MFI) with a 20-day period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=20).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=20).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))

    # Market Sentiment - Calculate the ratio of high to low prices, using a 10-day moving average for smoothing
    sentiment = (df['high'] / df['low']).rolling(window=10).mean()

    # Price-Volume Interaction
    price_volume_interaction = df['close'] * df['volume']

    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction
    return alpha_factor
