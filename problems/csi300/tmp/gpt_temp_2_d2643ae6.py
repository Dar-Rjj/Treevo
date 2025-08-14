import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum calculation - 100 period return for a longer-term trend
    momentum = df['close'].pct_change(100)
    
    # Liquidity - Calculate the average volume over a 60 day period to smooth out short-term fluctuations
    liquidity = df['volume'].rolling(window=60).mean()
    
    # Volatility - Calculate the rolling standard deviation of daily returns over a 30 day period for a more stable measure
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=30).std()
    
    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).rolling(window=30).mean()  # Use rolling mean for TR
    
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
    
    # Seasonality - Calculate the day of the week and month, and use it as a feature
    day_of_week = df.index.dayofweek
    month = df.index.month
    seasonality = (day_of_week + month) / 31  # Normalize to a range [0, 1]
    
    # Macroeconomic Indicator - Assuming we have an additional column 'macro' in the DataFrame
    macro_indicator = df['macro']
    
    # Technical Pattern - Moving Average Convergence Divergence (MACD)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction * (1 + seasonality) * (1 + macro_indicator) * macd_histogram
    return alpha_factor
