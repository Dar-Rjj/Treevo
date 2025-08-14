import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum calculation - adaptive window (50-100 period return)
    adaptive_momentum = df['close'].pct_change(rolling_window('close', 50, 100, df))
    
    # Liquidity - Calculate the average volume over an adaptive window (30-60 days)
    adaptive_liquidity = df['volume'].rolling(window=rolling_window('volume', 30, 60, df)).mean()
    
    # Volatility - Calculate the standard deviation of daily returns over an adaptive window (20-40 days)
    daily_returns = df['close'].pct_change()
    adaptive_volatility = daily_returns.rolling(window=rolling_window('close', 20, 40, df)).std()
    
    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = df[['high', 'low']].apply(lambda x: (x[0] - x[1]), axis=1)
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3) / 3
    
    # Money Flow Index (MFI) with an adaptive window (14-28 days)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=rolling_window('volume', 14, 28, df)).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=rolling_window('volume', 14, 28, df)).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Market Sentiment - Calculate the ratio of high to low prices
    sentiment = df['high'] / df['low']
    
    # Price-Volume Interaction
    price_volume_interaction = (df['close'] - df['close'].shift(1)) * df['volume']
    
    # Composite alpha factor
    alpha_factor = (adaptive_momentum * adaptive_liquidity / (adaptive_volatility + 1e-7)) * (mfi / 100) * (sentiment - 1) * price_volume_interaction
    return alpha_factor

def rolling_window(column, min_window, max_window, df):
    signal = df[column].pct_change().rolling(window=max_window).std()
    adaptive_window = (signal.rank(pct=True) * (max_window - min_window) + min_window).astype(int)
    return adaptive_window
