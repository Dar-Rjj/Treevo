import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the 30-day and 90-day simple moving average of close prices
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['sma_90'] = df['close'].rolling(window=90).mean()

    # Calculate the ratio of the 30-day to 90-day SMA, representing a medium to long-term trend
    sma_ratio = df['sma_30'] / df['sma_90']

    # Calculate the 7-day volume-weighted average price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).rolling(window=7).sum() / df['volume'].rolling(window=7).sum()

    # Calculate the relative change in VWAP over the past 14 days
    vwap_change = df['vwap'].pct_change(periods=14)

    # Calculate the 60-day high and low prices
    df['high_60'] = df['high'].rolling(window=60).max()
    df['low_60'] = df['low'].rolling(window=60).min()

    # Calculate the percentage distance from the 60-day high and low
    pct_from_high = (df['close'] - df['high_60']) / df['high_60']
    pct_from_low = (df['close'] - df['low_60']) / df['low_60']

    # Incorporate longer-term trends using exponential smoothing
    trend = df['close'].ewm(span=20, adjust=False).mean()

    # Calculate the difference between close and open price
    price_diff = df['close'] - df['open']

    # Calculate the average of high and low prices
    avg_price = (df['high'] + df['low']) / 2

    # Calculate the price volatility using the standard deviation of the price difference
    price_volatility = price_diff.rolling(window=10).std()

    # Calculate the momentum factor using the 12-day and 26-day Exponential Moving Averages (EMAs)
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    momentum = df['ema_12'] - df['ema_26']

    # Calculate the mean reversion factor using the 50-day and 200-day Simple Moving Averages (SMAs)
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    mean_reversion = (df['close'] - df['sma_200']) / (df['sma_50'] - df['sma_200'])

    # Calculate the adaptive volatility measure using the GARCH(1,1) model
    import arch
    garch_model = arch.arch_model(df['return'], vol='Garch', p=1, q=1)
    garch_res = garch_model.fit(disp='off')
    garch_volatility = garch_res.conditional_volatility

    # Combine the SMA ratio, VWAP change, percentage distances, trend, price movement, volatility, momentum, mean reversion, and adaptive volatility into a single factor
    factor = (sma_ratio + vwap_change + pct_from_high + pct_from_low + trend + (price_diff / avg_price) / (price_volatility + 1e-7) + momentum + mean_reversion + garch_volatility) / 9

    return factor
