import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Momentum - Using a dynamic window for momentum calculation based on the current market trend
    short_momentum = df['close'].pct_change(50)
    long_momentum = df['close'].pct_change(100)
    adaptive_momentum_window = (df['close'] > df['close'].shift(1)).rolling(window=30).mean() * 50 + (df['close'] < df['close'].shift(1)).rolling(window=30).mean() * 100
    momentum = (short_momentum.rolling(window=adaptive_momentum_window.astype(int), min_periods=1).mean() + long_momentum.rolling(window=adaptive_momentum_window.astype(int), min_periods=1).mean()) / 2
    
    # Liquidity - Calculate the average volume over a 60 day period to smooth out short-term fluctuations
    liquidity = df['volume'].rolling(window=60).mean()
    
    # Volatility - Calculate the rolling standard deviation of daily log returns over a 30 day period for a more stable measure
    daily_log_returns = np.log(df['close'] / df['close'].shift(1))
    volatility = daily_log_returns.rolling(window=30).std()
    
    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).rolling(window=30).mean()  # Use rolling mean for TR
    
    # Money Flow Index (MFI) with a 20-day period, adjusted for market dynamics
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=20).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=20).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Market Sentiment - Calculate the ratio of high to low prices, using a 10-day moving average for smoothing
    sentiment = (df['high'] / df['low']).rolling(window=10).mean()
    
    # Price-Volume Interaction
    price_volume_interaction = df['close'] * df['volume']
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction
    return alpha_factor
