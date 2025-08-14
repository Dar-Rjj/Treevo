import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the exponential moving average of close prices for a 5-day, 20-day, and 60-day window
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    # Calculate the momentum adjusted by volume for different EMA periods
    momentum_adjusted_5 = (df['close'] - ema_5) / df['volume']
    momentum_adjusted_20 = (df['close'] - ema_20) / df['volume']
    momentum_adjusted_60 = (df['close'] - ema_60) / df['volume']
    
    # Calculate the money flow over a 3-day, 10-day, and 30-day window
    money_flow_3 = (df['amount'] / df['volume']).rolling(window=3).mean()
    money_flow_10 = (df['amount'] / df['volume']).rolling(window=10).mean()
    money_flow_30 = (df['amount'] / df['volume']).rolling(window=30).mean()
    
    # Calculate the ratio of the current day's range to the 5-day, 20-day, and 60-day average range
    daily_range = df['high'] - df['low']
    avg_daily_range_5 = daily_range.rolling(window=5).mean()
    avg_daily_range_20 = daily_range.rolling(window=20).mean()
    avg_daily_range_60 = daily_range.rolling(window=60).mean()
    range_ratio_5 = daily_range / avg_daily_range_5
    range_ratio_20 = daily_range / avg_daily_range_20
    range_ratio_60 = daily_range / avg_daily_range_60
    
    # Adaptive volatility: use the standard deviation of the last 10 days, 30 days, and 60 days
    volatility_10 = df['close'].rolling(window=10).std()
    volatility_30 = df['close'].rolling(window=30).std()
    volatility_60 = df['close'].rolling(window=60).std()
    adaptive_volatility = (volatility_10 + volatility_30 + volatility_60) / 3
    
    # Additional feature: market sentiment (e.g., percentage change in volume)
    volume_change = df['volume'].pct_change().fillna(0)
    
    # Incorporate logarithmic returns as a measure of robust, responsive signals
