import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Momentum - Use exponential weighted average to give more weight to recent data
    short_momentum = df['close'].pct_change(50).ewm(span=50, adjust=False).mean()
    long_momentum = df['close'].pct_change(100).ewm(span=100, adjust=False).mean()
    momentum = (short_momentum + long_momentum) / 2

    # Liquidity - Calculate the exponential weighted moving average of volume over a 60 day period
    liquidity = df['volume'].ewm(span=60, adjust=False).mean()

    # Volatility - Calculate the exponential weighted standard deviation of daily returns for a responsive measure
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.ewm(span=30, adjust=False).std()

    # True Range (TR) calculation for volatility using exponential weighting
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).ewm(span=30, adjust=False).mean()

    # Money Flow Index (MFI) with a 20-day exponential weighted moving average
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).ewm(span=20, adjust=False).mean()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).ewm(span=20, adjust=False).mean()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))

    # Market Sentiment - Calculate the ratio of high to low prices, using an exponential weighted moving average for smoothing
    sentiment = (df['high'] / df['low']).ewm(span=10, adjust=False).mean()

    # Price-Volume Interaction
    price_volume_interaction = (df['close'] * df['volume']).ewm(span=60, adjust=False).mean()

    # Sector or Macroeconomic Indicator - Incorporate a hypothetical sector-specific indicator
    # For example, if we have a sector indicator 'sector_strength' in the DataFrame
    sector_strength = df['sector_strength'].ewm(span=60, adjust=False).mean()

    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction * sector_strength
    return alpha_factor
