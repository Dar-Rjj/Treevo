import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate the 5-period rolling mean of the daily return for momentum
    df['momentum_5'] = df['daily_return'].rolling(window=5).mean()
    
    # Calculate the change in amount and volume
    df['amount_change'] = df['amount'].diff()
    df['volume_change'] = df['volume'].diff()
    
    # Calculate the price change
    df['price_change'] = df['close'].diff()
    
    # Calculate the 5-period rolling standard deviation of the price change for volatility
    df['volatility_5'] = df['price_change'].rolling(window=5).std()
    
    # Calculate a factor that considers the interaction between momentum, price change, and volume change
    df['factor'] = (df['momentum_5'] * df['price_change'] * df['volume_change']) / (df['volatility_5'] + 1e-7)
    
    # Apply a 5-period rolling mean to the factor for smoothing
    df['factor_smoothed'] = df['factor'].rolling(window=5).mean()
    
    # Calculate the range of the day
    df['day_range'] = df['high'] - df['low']
    
    # Calculate the average price
    df['avg_price'] = (df['open'] + df['close']) / 2
    
    # Calculate the weighted volume by the average price
    df['weighted_volume'] = df['volume'] * df['avg_price']
    
    # Calculate the ratio of today's weighted volume to the moving average of the weighted volume over a certain period
    df['moving_avg_weighted_volume'] = (df['volume'] * df['close']).rolling(window=5).mean()
    df['volume_ratio'] = df['weighted_volume'] / (df['moving_avg_weighted_volume'] + 1e-7)
    
    # Calculate the 20-day simple moving average
    df['sma_20d'] = df['close'].rolling(window=20).mean()
    
    # Calculate the trend factor using the 5-day and 20-day simple moving average
    df['trend_factor'] = (df['sma_5d'] - df['sma_20d']) / df['close']
    
    # Combine the factors: smoothed factor, day range, volume ratio, and trend
    df['combined_factor'] = df['factor_smoothed'] * df['day_range'] * df['volume_ratio'] * df['trend_factor']
    
    return df['combined_factor']
