import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the daily return
    daily_return = df['close'].pct_change()
    
    # Calculate the 5-day and 20-day Exponentially Weighted Moving Average (EWMA) of the returns for momentum
    ewma_momentum_5 = daily_return.ewm(span=5, adjust=False).mean()
    ewma_momentum_20 = daily_return.ewm(span=20, adjust=False).mean()
    
    # Calculate the 10-day and 30-day Exponentially Weighted Moving Standard Deviation (EWMSTD) of the returns for volatility
    ewmstd_volatility_10 = daily_return.ewm(span=10, adjust=False).std()
    ewmstd_volatility_30 = daily_return.ewm(span=30, adjust=False).std()
    
    # Calculate the 5-day and 20-day Exponentially Weighted Moving Average (EWMA) of the volume for liquidity
    ewma_volume_5 = df['volume'].ewm(span=5, adjust=False).mean()
    ewma_volume_20 = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Calculate cumulative sum of the daily returns for market sentiment
    cumulative_return = daily_return.cumsum()
    
    # Calculate the factor as a combination of momentum, volatility, and liquidity
    alpha_factor = (ewma_momentum_5 / (ewmstd_volatility_10 + 1e-7)) * (1 / (ewma_volume_5 + 1e-7)) \
                  + (ewma_momentum_20 / (ewmstd_volatility_30 + 1e-7)) * (1 / (ewma_volume_20 + 1e-7)) \
                  + cumulative_return
    
    # Calculate the 5-day and 20-day Exponentially Weighted Moving Average (EWMA) of the high and low prices for price stability
    ewma_high_5 = df['high'].ewm(span=5, adjust=False).mean()
    ewma_low_5 = df['low'].ewm(span=5, adjust=False).mean()
    ewma_high_20 = df['high'].ewm(span=20, adjust=False).mean()
    ewma_low_20 = df['low'].ewm(span=20, adjust=False).mean()
    
    # Calculate the price range stability
    price_stability_5 = (ewma_high_5 - ewma_low_5) / (ewma_high_5 + ewma_low_5 + 1e-7)
    price_stability_20 = (ewma_high_20 - ewma_low_20) / (ewma_high_20 + ewma_low_20 + 1e-7)
    
    # Incorporate price stability into the alpha factor
    alpha_factor += (price_stability_5 - price_stability_20)
    
    # Add a volume-weighted average price (VWAP) component
    vwap = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
    alpha_factor += (vwap / df['close'] - 1)
    
    # Incorporate a longer-term trend by using a 60-day EWMA of the close price
    ewma_close_60 = df['close'].ewm(span=60, adjust=False).mean()
    long_term_trend = df['close'] / ewma_close_60 - 1
    alpha_factor += long_term_trend
    
    # Calculate the dynamic window based on the rolling standard deviation of the returns
    rolling_std_10 = daily_return.rolling(window=10).std()
    dynamic_window = (rolling_std_10 > rolling_std_10.shift(1)).astype(int)
    
    # Adjust the alpha factor with the dynamic window
    alpha_factor *= dynamic_window
    
    # Adaptive window for EWM based on the rolling standard deviation
    adaptive_span = 5 + (rolling_std_10 > 0.01).astype(int) * 15
    adaptive_ewma_momentum = daily_return.ewm(span=adaptive_span, adjust=False).mean()
    alpha_factor += adaptive_ewma_momentum
    
    return alpha_factor
