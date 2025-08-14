import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the difference between today's close and yesterday's close, then divide by today's volume
    momentum_adjusted = (df['close'].diff() / df['volume']).shift(1)
    
    # Calculate the weighted price using amount and volume to identify the money flow
    money_flow = (df['amount'] / df['volume']).rolling(window=3).mean()
    
    # Calculate the ratio of the current day's range to the 5-day average range
    daily_range = df['high'] - df['low']
    avg_daily_range = daily_range.rolling(window=5).mean()
    range_ratio = daily_range / avg_daily_range
    
    # Incorporate dynamic volatility using the exponentially weighted standard deviation of the last 10 days' close prices
    volatility = df['close'].ewm(span=10, adjust=False).std()
    
    # Calculate the exponential moving average (EMA) of the close price for short (5) and long (20) periods
    close_ema_short = df['close'].ewm(span=5, adjust=False).mean()
    close_ema_long = df['close'].ewm(span=20, adjust=False).mean()
    
    # Compute the trend strength as the difference between the short and long EMA
    trend_strength = (close_ema_short - close_ema_long) / df['close']
    
    # Use adaptive windows for smoothing
    adaptive_window = 10 * (volatility / volatility.ewm(span=10, adjust=False).mean())
    adaptive_window = adaptive_window.apply(lambda x: int(x) if not pd.isna(x) else 5)
    
    # Combine all factors with a weighted sum, where the weights are based on the adaptive window
    alpha_factor = (momentum_adjusted + money_flow + range_ratio + trend_strength + volatility) / 5
    alpha_factor_smoothed = alpha_factor.rolling(window=adaptive_window, min_periods=5).mean()
    
    # Apply exponential smoothing to the combined factor
    smoothed_alpha_factor = alpha_factor_smoothed.ewm(span=5, adjust=False).mean()
    
    return smoothed_alpha_factor
