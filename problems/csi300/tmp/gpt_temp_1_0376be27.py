import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Momentum - Use a dynamic window size based on the ATR to capture market trends
    short_momentum_window = 50 + (df['close'].pct_change().abs() * 10).rolling(window=50).mean().fillna(50).astype(int)
    long_momentum_window = 100 + (df['close'].pct_change().abs() * 10).rolling(window=100).mean().fillna(100).astype(int)
    short_momentum = df['close'].pct_change(short_momentum_window).ewm(span=short_momentum_window, adjust=False).mean()
    long_momentum = df['close'].pct_change(long_momentum_window).ewm(span=long_momentum_window, adjust=False).mean()
    momentum = (short_momentum + long_momentum) / 2

    # Liquidity - Calculate the average volume over a 60 day period to smooth out short-term fluctuations
    liquidity = df['volume'].rolling(window=60).mean()

    # Volatility - Calculate the rolling standard deviation of daily returns over a 30 day period for a more stable measure
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=30).std()

    # True Range (TR) calculation for adaptive volatility
    prev_close = df['close'].shift(1)
    tr1 = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr1 + tr2 + tr3).rolling(window=30).mean()  # Use rolling mean for TR

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

    # Non-linear transformations
    log_momentum = np.log1p(momentum)
    sqrt_liquidity = np.sqrt(liquidity)
    inv_volatility = 1 / (volatility + 1e-7)

    # Composite alpha factor
    alpha_factor = (log_momentum * sqrt_liquidity * inv_volatility) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction
    return alpha_factor
