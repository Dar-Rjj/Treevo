import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum calculation - 50 period log return
    momentum = (df['close'].pct_change(50) + 1).apply(np.log)
    
    # Adaptive Rolling Window for Liquidity - Calculate the weighted average volume over a 30 day period
    weights = np.exp(np.linspace(-1., 0., 30))
    liquidity = df['volume'].rolling(window=30).apply(lambda x: np.average(x, weights=weights), raw=True)
    
    # Volatility - Calculate the standard deviation of daily log returns over a 20 day period
    daily_log_returns = (df['close'].pct_change() + 1).apply(np.log)
    volatility = daily_log_returns.rolling(window=20).std()
    
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
    
    # Macro Data Integration - Assuming a column 'macro_data' is available in the DataFrame
    macro_factor = df['macro_data']
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (sentiment - 1) * macro_factor
    return alpha_factor
