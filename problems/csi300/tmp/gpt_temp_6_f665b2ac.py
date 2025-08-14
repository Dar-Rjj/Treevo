import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the net change for the day
    net_change = df['close'] - df['open']
    
    # Calculate the daily range (high - low)
    daily_range = df['high'] - df['low']
    
    # Calculate the smoothed volume using an adaptive exponential moving average
    smoothed_volume = df['volume'].ewm(span=df['volume'].rolling(window=5).mean()).mean()
    
    # Calculate the momentum as the 5-day return
    momentum = df['close'].pct_change(periods=5)
    
    # Calculate the realized volatility as the standard deviation of the last 5 days' returns
    volatility = df['close'].pct_change().rolling(window=5).std()
    
    # Calculate a liquidity measure as the ratio of volume to the range
    liquidity = df['volume'] / (daily_range + 1e-7)
    
    # Calculate the 20-day simple moving average
    sma_20 = df['close'].rolling(window=20).mean()
    
    # Calculate the distance from the 20-day SMA as a mean reversion signal
    mean_reversion_signal = (df['close'] - sma_20) / (sma_20 + 1e-7)
    
    # Calculate the 50-day simple moving average
    sma_50 = df['close'].rolling(window=50).mean()
    
    # Calculate the trend signal as the difference between the 50-day SMA and the 20-day SMA
    trend_signal = (sma_50 - sma_20) / (sma_20 + 1e-7)
    
    # Incorporate macroeconomic data, such as the VIX index, if available
    if 'vix' in df.columns:
        vix_adjustment = 1 / (df['vix'] + 1e-7)
    else:
        vix_adjustment = 1.0  # Default value if VIX is not provided
    
    # The factor is a combination of the standardized net change, adjusted by the daily range, smoothed volume, momentum, volatility, mean reversion, trend, and macroeconomic data
    factor_value = (
        (net_change / (daily_range + 1e-7)) * 
        (smoothed_volume / df['volume']) * 
        momentum * 
        (1 / (volatility + 1e-7)) * 
        liquidity * 
        (1 - mean_reversion_signal) * 
        trend_signal * 
        vix_adjustment
    )

    return factor_value
