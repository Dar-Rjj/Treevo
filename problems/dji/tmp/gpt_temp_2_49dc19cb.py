import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate daily log return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Compute 5-day and 20-day simple moving averages (SMA)
    df['5_day_SMA'] = df['close'].rolling(window=5).mean()
    df['20_day_SMA'] = df['close'].rolling(window=20).mean()
    
    # Subtract SMAs from the current day's close price
    df['close_minus_5_day_SMA'] = df['close'] - df['5_day_SMA']
    df['close_minus_20_day_SMA'] = df['close'] - df['20_day_SMA']
    
    # Compute the difference between the 5-day and 20-day SMAs
    df['SMA_diff'] = df['5_day_SMA'] - df['20_day_SMA']
    
    # Calculate daily change in trading volume
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Combine volume change with price movement
    df['log_return_volume_change'] = df['log_return'] * df['volume_change']
    
    # Calculate 5-day moving average of volume
    df['5_day_vol_SMA'] = df['volume'].rolling(window=5).mean()
    
    # Identify unusual volume spikes
    df['volume_minus_5_day_vol_SMA'] = df['volume'] - df['5_day_vol_SMA']
    
    # Develop a factor based on candlestick formation
    df['body_length'] = abs(df['open'] - df['close'])
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['candlestick_factor'] = df['body_length'] - (df['upper_shadow'] + df['lower_shadow'])
    
    # Compute the percentage of the body relative to the total range (high - low)
    df['body_percentage'] = df['body_length'] / (df['high'] - df['low'])
    
    # Compare today's transaction amount to a short-term (e.g., 5 days) moving average
    df['5_day_amount_SMA'] = df['amount'].rolling(window=5).mean()
    df['amount_minus_5_day_amount_SMA'] = df['amount'] - df['5_day_amount_SMA']
    
    # Create a weighted sum of selected sub-thought indicators
    weights = {
        'log_return': 0.2,
        'close_minus_5_day_SMA': 0.1,
        'close_minus_20_day_SMA': 0.1,
        'SMA_diff': 0.1,
        'log_return_volume_change': 0.1,
        'volume_minus_5_day_vol_SMA': 0.1,
        'candlestick_factor': 0.1,
        'body_percentage': 0.1,
        'amount_minus_5_day_amount_SMA': 0.1
    }
    df['composite_factor'] = (df[list(weights.keys())] * pd.Series(weights)).sum(axis=1)
    
    # Introduce a machine learning-based composite factor
    X = df[['log_return', 'close_minus_5_day_SMA', 'close_minus_20_day_SMA', 'SMA_diff', 
            'log_return_volume_change', 'volume_minus_5_day_vol_SMA', 'candlestick_factor', 
            'body_percentage', 'amount_minus_5_day_amount_SMA']].dropna()
    y = df['log_return'].shift(-1).loc[X.index]  # Use next day's log return as target
    model = LinearRegression()
    model.fit(X, y)
    df['ml_composite_factor'] = model.predict(df[['log_return', 'close_minus_5_day_SMA', 
                                                   'close_minus_20_day_SMA', 'SMA_diff', 
                                                   'log_return_volume_change', 'volume_minus_5_day_vol_SMA', 
                                                   'candlestick_factor', 'body_percentage', 
                                                   'amount_minus_5_day_amount_SMA']])
    
    # Return the final alpha factor
    return df['composite_factor'] + df['ml_composite_factor']

# Example usage:
# df = pd.read_csv('your_data.csv')
# df.set_index('date', inplace=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
