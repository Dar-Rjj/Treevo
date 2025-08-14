import numpy as np
def heuristics_v2(df):
    # Calculate the Exponential Moving Average (EMA) of closing prices
    ema_period = 50
    df['EMA'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    
    # Generate a signal; 1 if the current close is above the EMA, -1 if below, 0 if equal
    df['trend_signal'] = np.where(df['close'] > df['EMA'], 1, np.where(df['close'] < df['EMA'], -1, 0))
    
    # Calculate the difference between today’s high and the previous day’s high
    df['high_diff'] = df['high'] - df['high'].shift(1)
    
    # Calculate the difference between today’s low and the previous day’s low
    df['low_diff'] = df['low'] - df['low'].shift(1)
    
    # Sum the differences to get a combined momentum score
    df['momentum_score'] = df['high_diff'] + df['low_diff']
    
    # Create a simple threshold to classify stocks as having positive or negative momentum based on the score
    df['momentum_signal'] = np.where(df['momentum_score'] > 0, 1, -1)
    
    # Evaluate the percentage change in volume from the previous day
    df['volume_change'] = df['volume'].pct_change()
    
    # Assign a value indicating strong buying (positive change) or selling (negative change) pressure
    df['volume_signal'] = np.where(df['volume_change'] > 0, 1, -1)
    
    # Multiply the return (close to close) by the square root of the volume traded
