import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Momentum - 50 to 100 period return based on the current market trend
    short_momentum = df['close'].pct_change(50)
    long_momentum = df['close'].pct_change(100)
    momentum = (short_momentum + long_momentum) / 2
    
    # Liquidity - Calculate the average volume over a dynamic window (60 to 120 days) based on recent volatility
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=30).std()
    liquidity_window = (volatility * 60).astype(int).clip(lower=60, upper=120)
    liquidity = df['volume'].rolling(window=liquidity_window).mean()
    
    # Volatility - Exponential Moving Average (EMA) of daily returns over a 30 day period
    ema_volatility = daily_returns.ewm(span=30, adjust=False).std()
    
    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).ewm(span=30, adjust=False).mean()  # Use EMA for TR
    
    # Money Flow Index (MFI) with a 20-day period, adjusted for sensitivity
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=20).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=20).sum()
    mfi = 100 - (100 / (1 + (positive_money_flow / (negative_money_flow + 1e-7)) ** 0.5))
    
    # Market Sentiment - Calculate the ratio of high to low prices, using a 10-day EMA for smoothing
    sentiment = (df['high'] / df['low']).ewm(span=10, adjust=False).mean()
    
    # Price-Volume Interaction
    price_volume_interaction = df['close'] * df['volume']
    
    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (ema_volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction
    return alpha_factor
