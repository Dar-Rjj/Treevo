import pandas as pd
def heuristics_v2(df: pd.DataFrame, macroeconomic_data: pd.DataFrame) -> pd.Series:
    # Calculate the simple moving average of close prices for a 5-day window
    sma_5 = df['close'].rolling(window=5).mean()
    
    # Calculate the difference between today's close and the 5-day SMA, then divide by today's volume
    momentum_adjusted = (df['close'] - sma_5) / df['volume']
    
    # Calculate the weighted price using amount and volume to identify the money flow over a 3-day window
    money_flow = (df['amount'] / df['volume']).rolling(window=3).mean()
    
    # Calculate the ratio of the current day's range to the 5-day average range
    daily_range = df['high'] - df['low']
    avg_daily_range = daily_range.rolling(window=5).mean()
    range_ratio = daily_range / avg_daily_range
    
    # Adaptive window for volatility: use the standard deviation of the last 10 days
    volatility = df['close'].rolling(window=10).std()
    
    # Calculate the adaptive weight for volatility
    adaptive_weight = 1 / (1 + volatility)
    
    # Incorporate seasonality (e.g., monthly or quarterly patterns)
    month = df.index.month
    seasonal_factor = pd.get_dummies(month, prefix='month')
    df = pd.concat([df, seasonal_factor], axis=1)
    
    # Use macroeconomic data for additional indicators
    merged_df = df.join(macroeconomic_data, how='left').fillna(method='ffill')
    
    # Calculate a combined macroeconomic factor
    macro_factor = (merged_df['inflation_rate'] + merged_df['interest_rate']) / 2
    
    # Combine all factors with a weighted sum, where the weights are based on the adaptive volatility
    alpha_factor = (momentum_adjusted * adaptive_weight + 
                    0.5 * money_flow * adaptive_weight + 
                    0.3 * range_ratio * adaptive_weight + 
                    0.2 * macro_factor * adaptive_weight)
    
    # Add seasonality to the alpha factor
    alpha_factor += seasonal_factor.sum(axis=1) * 0.1
    
    # Smooth the alpha factor using an exponential moving average with a span of 5
    alpha_factor_smoothed = alpha_factor.ewm(span=5, adjust=False).mean()
    
    return alpha_factor_smoothed
