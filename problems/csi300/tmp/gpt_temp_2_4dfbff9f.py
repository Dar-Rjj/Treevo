import pandas as pd
def heuristics_v2(df: pd.DataFrame, macroeconomic_indicators: pd.DataFrame) -> pd.Series:
    # Momentum calculation - 50 and 20 period return
    momentum_50 = df['close'].pct_change(50)
    momentum_20 = df['close'].pct_change(20)

    # Liquidity - Calculate the average volume over a 30 day and 10 day period
    liquidity_30 = df['volume'].rolling(window=30).mean()
    liquidity_10 = df['volume'].rolling(window=10).mean()

    # Volatility - Calculate the standard deviation of daily returns over a 20 day and 10 day period
    daily_returns = df['close'].pct_change()
    volatility_20 = daily_returns.rolling(window=20).std()
    volatility_10 = daily_returns.rolling(window=10).std()

    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = df[['high', 'low']].apply(lambda x: (x[0] - x[1]), axis=1)
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3) / 3

    # Money Flow Index (MFI) with a 14-day period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))

    # Market Sentiment - Calculate the ratio of high to low prices
    sentiment = df['high'] / df['low']

    # Price-Volume Interaction
    price_volume_interaction = df['close'] * df['volume']

    # Incorporate macroeconomic indicators
    inflation_rate = macroeconomic_indicators['inflation_rate']
    gdp_growth = macroeconomic_indicators['gdp_growth']
    unemployment_rate = macroeconomic_indicators['unemployment_rate']

    # Composite alpha factor
    alpha_factor = (momentum_50 + momentum_20) * (liquidity_30 + liquidity_10) / (volatility_20 + volatility_10 + 1e-7) * (mfi / 100) * (sentiment - 1) * price_volume_interaction
    alpha_factor *= (1 + gdp_growth - inflation_rate - unemployment_rate)

    return alpha_factor
