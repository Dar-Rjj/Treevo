import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the 5-day and 20-day exponential moving average of closing prices to capture smooth trends
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Momentum factor: difference between 5-day and 20-day EMA
    df['momentum_factor'] = df['EMA_5'] - df['EMA_20']

    # Advanced volatility measure: GARCH(1,1) model on returns
    from arch import arch_model
    returns = df['close'].pct_change().dropna()
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_res = garch_model.fit(disp='off')
    df['garch_volatility'] = garch_res.conditional_volatility.reindex_like(df, method='bfill')

    # Volume-weighted average price (VWAP) over 5 days to capture liquidity
    df['VWAP_5'] = ((df['close'] * df['volume']).rolling(window=5).sum()) / df['volume'].rolling(window=5).sum()
    df['liquidity_factor'] = df['VWAP_5']

    # Relative strength as the ratio of current close to 20-day EMA
    df['relative_strength'] = df['close'] / df['EMA_20']

    # Market breadth: difference between 5-day and 20-day moving averages of (high - low)
    df['range_5'] = (df['high'] - df['low']).rolling(window=5).mean()
    df['range_20'] = (df['high'] - df['low']).rolling(window=20).mean()
    df['market_breadth'] = df['range_5'] - df['range_20']

    # Additional factor: Money Flow Index (MFI) over 14 days
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_money_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    MFI_14 = 100 - (100 / (1 + (positive_money_flow.rolling(window=14).sum() / negative_money_flow.rolling(window=14).sum())))
    df['MFI_14'] = MFI_14

    # Incorporate macroeconomic indicators (e.g., GDP growth rate, inflation rate)
    # For simplicity, assume these are provided in the DataFrame
    df['macro_indicator'] = df['GDP_growth_rate'] - df['inflation_rate']

    # Dynamic weights for the factors
    df['momentum_weight'] = df['garch_volatility'].rank(pct=True)
    df['liquidity_weight'] = df['liquidity_factor'].rank(pct=True)
    df['relative_strength_weight'] = df['relative_strength'].rank(pct=True)
    df['market_breadth_weight'] = df['market_breadth'].rank(pct=True)
    df['MFI_weight'] = df['MFI_14'].rank(pct=True)
    df['macro_weight'] = df['macro_indicator'].rank(pct=True)

    # Combine the factors with dynamic weights
    heuristic_factor = (df['momentum_factor'] * df['momentum_weight']) + \
                       (df['liquidity_factor'] * df['liquidity_weight']) + \
                       (df['relative_strength'] * df['relative_strength_weight']) + \
                       (df['market_breadth'] * df['market_breadth_weight']) + \
                       (df['MFI_14'] * df['MFI_weight']) + \
                       (df['macro_indicator'] * df['macro_weight'])

    return heuristic_factor
