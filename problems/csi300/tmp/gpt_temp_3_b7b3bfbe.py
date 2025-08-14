import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum calculation - 50 period return
    momentum = df['close'].pct_change(50)
    
    # Liquidity - Calculate the weighted average volume over a 30 day period with more weight on recent days
    weights = np.linspace(1, 2, 30)
    liquidity = df['volume'].rolling(window=30).apply(lambda x: np.sum(x * weights) / np.sum(weights), raw=True)
    
    # Volatility - Calculate the standard deviation of daily log returns over a 20 day period
    daily_log_returns = np.log(df['close'] / df['close'].shift(1))
    volatility = daily_log_returns.rolling(window=20).std()
    
    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = df[['high', 'low']].apply(lambda x: (x[0] - x[1]), axis=1)
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3) / 3
    
    # Money Flow Index (MFI) with an adaptive window based on the average true range
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    window = 14 + (avg_true_range * 10).astype(int)
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window, min_periods=1).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window, min_periods=1).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Market Sentiment - Calculate the ratio of high to low prices with a seasonal adjustment
    sentiment = (df['high'] / df['low']) * (1 + 0.01 * np.sin(2 * np.pi * df.index.dayofyear / 365))
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (sentiment - 1)
    return alpha_factor
