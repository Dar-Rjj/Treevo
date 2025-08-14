import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the exponential moving average of close prices for a 5-day and 20-day window
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate the difference between today's close and the EMA values, then divide by today's volume
    momentum_adjusted_5 = (df['close'] - ema_5) / df['volume']
    momentum_adjusted_20 = (df['close'] - ema_20) / df['volume']
    
    # Calculate the weighted price using amount and volume to identify the money flow over a 3-day and 10-day window
    money_flow_3 = (df['amount'] / df['volume']).rolling(window=3).mean()
    money_flow_10 = (df['amount'] / df['volume']).rolling(window=10).mean()
    
    # Adaptive window for volatility: use the standard deviation of the last 10 days and 30 days
    volatility_10 = df['close'].rolling(window=10).std()
    volatility_30 = df['close'].rolling(window=30).std()
    adaptive_volatility = (volatility_10 + volatility_30) / 2
    
    # Additional feature: market sentiment (e.g., percentage change in volume)
    volume_change = df['volume'].pct_change().fillna(0)
    
    # Incorporate logarithmic returns as a measure of robust, responsive signals
