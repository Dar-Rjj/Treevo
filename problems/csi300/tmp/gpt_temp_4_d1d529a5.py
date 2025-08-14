import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Dynamic window size for momentum based on the ATR
    atr_50 = (df['high'] - df['low']).abs().rolling(window=50).mean()
    atr_100 = (df['high'] - df['low']).abs().rolling(window=100).mean()
    dynamic_window = (atr_50 + atr_100) / 2
    short_momentum = df['close'].pct_change(dynamic_window.astype(int))
    long_momentum = df['close'].pct_change(dynamic_window.astype(int) * 2)
    momentum = (short_momentum + long_momentum) / 2

    # Liquidity - Combine rolling and EWM to smooth out short-term fluctuations and react to recent changes
    liquidity_rolling = df['volume'].rolling(window=60).mean()
    liquidity_ewm = df['volume'].ewm(span=60).mean()
    liquidity = (liquidity_rolling + liquidity_ewm) / 2

    # Volatility - Combine rolling and EWM standard deviation for a more stable measure
    daily_returns = df['close'].pct_change()
    volatility_rolling = daily_returns.rolling(window=30).std()
    volatility_ewm = daily_returns.ewm(span=30).std()
    volatility = (volatility_rolling + volatility_ewm) / 2

    # True Range (TR) calculation for adaptive volatility
    prev_close = df['close'].shift(1)
    tr1 = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr1 + tr2 + tr3).rolling(window=30).mean()  # Use rolling mean for TR

    # Money Flow Index (MFI) with a 20-day period, using a combination of rolling and EWM
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow_rolling = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=20).sum()
    negative_money_flow_rolling = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=20).sum()
    positive_money_flow_ewm = raw_money_flow.where(typical_price > typical_price.shift(1), 0).ewm(span=20).mean()
    negative_money_flow_ewm = raw_money_flow.where(typical_price < typical_price.shift(1), 0).ewm(span=20).mean()
    mfi_rolling = 100 - (100 / (1 + positive_money_flow_rolling / (negative_money_flow_rolling + 1e-7)))
    mfi_ewm = 100 - (100 / (1 + positive_money_flow_ewm / (negative_money_flow_ewm + 1e-7)))
    mfi = (mfi_rolling + mfi_ewm) / 2

    # Market Sentiment - Calculate the ratio of high to low prices, using a 10-day moving average for smoothing
    sentiment = (df['high'] / df['low']).rolling(window=10).mean()

    # Price-Volume Interaction
    price_volume_interaction = df['close'] * df['volume']

    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction
    return alpha_factor
