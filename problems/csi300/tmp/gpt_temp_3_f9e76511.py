import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum calculation - 50 period return
    momentum = df['close'].pct_change(50)
    
    # Liquidity - Calculate the average volume over a 30 day period with an adaptive rolling window
    window_size = df['volume'].rolling(window=30).std().rolling(window=10).mean() * 2
    liquidity = df['volume'].rolling(window=window_size.astype(int)).mean()
    
    # Volatility - Calculate the standard deviation of daily log returns over a 20 day period
    daily_returns = np.log(df['close']).diff()
    volatility = daily_returns.rolling(window=20).std()
    
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
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (sentiment - 1)
    
    # Adjust for market regimes - Example: Bullish vs Bearish
    regime = np.where(daily_returns.rolling(window=20).mean() > 0, 1, 0)
    alpha_factor_adjusted = alpha_factor * (regime * 1.5 + (1 - regime) * 0.5)
    
    return alpha_factor_adjusted
