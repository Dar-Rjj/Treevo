import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Dynamic Momentum - 50 to 100 period return based on the current market trend, with a dynamic window size
    short_momentum = df['close'].pct_change(50).ewm(span=50, adjust=False).mean()
    long_momentum = df['close'].pct_change(100).ewm(span=100, adjust=False).mean()
    momentum = (short_momentum + long_momentum) / 2
    
    # Liquidity - Calculate the exponential moving average volume over a 60 day period to smooth out short-term fluctuations
    liquidity = df['volume'].ewm(span=60, adjust=False).mean()
    
    # Robust Volatility - Calculate the exponential weighted standard deviation of daily log returns over a 30 day period for a more stable measure
    daily_log_returns = np.log(df['close'] / df['close'].shift(1))
    volatility = daily_log_returns.ewm(span=30, adjust=False).std()
    
    # True Range (TR) calculation for robust volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).ewm(span=30, adjust=False).mean()  # Use exponential weighted mean for TR
    
    # Money Flow Index (MFI) with a 20-day period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=20).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=20).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Market Sentiment - Calculate the ratio of high to low prices, using a 10-day exponential moving average for smoothing
    sentiment = (df['high'] / df['low']).ewm(span=10, adjust=False).mean()
    
    # Price-Volume Interaction
    price_volume_interaction = df['close'] * df['volume']
    
    # Macro Indicators - Integrate macroeconomic indicators for enhanced predictive power
    # Example: Using a hypothetical macro indicator 'macro_indicator' from the DataFrame
    macro_indicator = df['macro_indicator']  # Assume this column is present in the DataFrame
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction * macro_indicator
    return alpha_factor
