import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Trend Strength - Calculate the ADX (Average Directional Index) to measure trend strength
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['close'].shift(1) - df['low']).abs()
    true_range = pd.concat([tr, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    plus_dm = (df['high'].diff().clip(lower=0) - df['high'].shift(1).combine(df['low'], lambda x, y: 0 if x < y else 0).diff().clip(upper=0))
    minus_dm = (df['low'].shift(1).combine(df['high'], lambda x, y: 0 if x > y else 0).diff().clip(upper=0) - df['low'].diff().clip(lower=0))
    
    plus_di = 100 * (plus_dm.rolling(window=14).sum() / atr)
    minus_di = 100 * (minus_dm.abs().rolling(window=14).sum() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=14).mean()
    
    # Adaptive Momentum - 50 to 100 period return based on the current market trend
    short_momentum = df['close'].pct_change(50)
    long_momentum = df['close'].pct_change(100)
    momentum = (short_momentum + long_momentum) / 2
    
    # Liquidity - Calculate the exponential moving average of volume over a 60 day period to smooth out short-term fluctuations
    liquidity = df['volume'].ewm(span=60).mean()
    
    # Volatility - Calculate the exponential moving standard deviation of daily returns over a 30 day period for a more stable measure
    daily_returns = df['close'].pct_change().apply(np.log1p)
    volatility = daily_returns.ewm(span=30).std()
    
    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).ewm(span=30).mean()  # Use exponential moving mean for TR
    
    # Money Flow Index (MFI) with a 20-day period, adjusting sensitivity
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).ewm(span=20).mean()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).ewm(span=20).mean()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))
    
    # Market Sentiment - Calculate the ratio of high to low prices, using a 10-day exponential moving average for smoothing
    sentiment = (df['high'] / df['low']).ewm(span=10).mean()
    
    # Price-Volume Interaction
    price_volume_interaction = df['close'] * df['volume']
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction
    alpha_factor = alpha_factor * (adx / 100)  # Incorporate ADX for trend strength
    
    return alpha_factor
