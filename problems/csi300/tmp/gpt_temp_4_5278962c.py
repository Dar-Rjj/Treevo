import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Momentum - Dynamic period return based on the moving average of the last 50 and 100 day returns
    short_momentum = df['close'].pct_change(50).ewm(span=20, adjust=False).mean()
    long_momentum = df['close'].pct_change(100).ewm(span=20, adjust=False).mean()
    momentum = (short_momentum + long_momentum) / 2
    
    # Liquidity - Calculate the exponentially smoothed average volume over a 60 day period
    liquidity = df['volume'].ewm(span=60, adjust=False).mean()
    
    # Volatility - Calculate the exponentially smoothed standard deviation of daily log returns over a 30 day period
    daily_log_returns = np.log(df['close'] / df['close'].shift(1))
    volatility = daily_log_returns.ewm(span=30, adjust=False).std()
    
    # True Range (TR) calculation for volatility using exponential smoothing
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).ewm(span=30, adjust=False).mean()
    
    # Money Flow Index (MFI) with a 20-day exponential moving average period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).ewm(span=20, adjust=False).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).ewm(span=20, adjust=False).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Market Sentiment - Exponentially smoothed ratio of high to low prices over a 10-day period
    sentiment = (df['high'] / df['low']).ewm(span=10, adjust=False).mean()
    
    # Price-Volume Interaction
    price_volume_interaction = (df['close'] * df['volume']).ewm(span=20, adjust=False).mean()
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction
    return alpha_factor
