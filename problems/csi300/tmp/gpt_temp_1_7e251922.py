import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the adaptive exponential moving average of close prices
    alpha = 2 / (df['close'].expanding().std() + 1)
    ema_close = df['close'].ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate the momentum adjusted by volume
    momentum_adjusted = (df['close'] - ema_close) / df['volume']
    
    # Calculate the volume-adjusted money flow over a 3-day window
    money_flow = (df['amount'] / df['volume']).rolling(window=3).mean()
    
    # Calculate the ratio of the current day's range to the 5-day average range
    daily_range = df['high'] - df['low']
    avg_daily_range = daily_range.rolling(window=5).mean()
    range_ratio = daily_range / avg_daily_range
    
    # Adaptive volatility: use the standard deviation of the last 10 days
    volatility = df['close'].rolling(window=10).std()
    
    # Combine all factors with a weighted sum, where the weights are based on the adaptive volatility
    alpha_factor = (momentum_adjusted + 0.5 * money_flow + 0.3 * range_ratio) / (1 + volatility)
    
    # Smooth the alpha factor using an exponential moving average with a span of 5
    alpha_factor_smoothed = alpha_factor.ewm(span=5, adjust=False).mean()
    
    return alpha_factor_smoothed
