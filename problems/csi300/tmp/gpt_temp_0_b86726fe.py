import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Momentum - Dynamic adjustment based on the 50 and 100 period returns
    short_momentum = df['close'].pct_change(50)
    long_momentum = df['close'].pct_change(100)
    momentum = (short_momentum + long_momentum) / 2
    
    # Liquidity - Average volume over a 60 day period
    liquidity = df['volume'].rolling(window=60).mean()
    
    # Volatility - Rolling standard deviation of daily returns over a 30 day period
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=30).std()
    
    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).rolling(window=30).mean()
    
    # Money Flow Index (MFI) with a 20-day period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=20).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=20).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Market Sentiment - High to low price ratio, 10-day moving average
    sentiment = (df['high'] / df['low']).rolling(window=10).mean()
    
    # Price-Volume Interaction
    price_volume_interaction = df['close'] * df['volume']
    
    # Trend Strength - Calculate the slope of the 50-day moving average
    trend_strength = df['close'].rolling(window=50).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction * trend_strength
    return alpha_factor
