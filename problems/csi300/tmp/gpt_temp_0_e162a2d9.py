import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Raw Returns
    df['returns'] = df['close'].pct_change()
    
    # Compute 14-Day Sum of Upward Returns
    df['up_returns'] = df['returns'].apply(lambda x: x if x > 0 else 0)
    df['up_sum_14'] = df['up_returns'].rolling(window=14).sum()
    
    # Compute 14-Day Sum of Downward Returns
    df['down_returns'] = df['returns'].apply(lambda x: -x if x < 0 else 0)
    df['down_sum_14'] = df['down_returns'].rolling(window=14).sum()
    
    # Calculate Relative Strength
    df['rs'] = df['up_sum_14'] / df['down_sum_14']
    
    # Smooth with Exponential Moving Average on Volume
    df['ema_volume_14'] = df['volume'].ewm(span=14, adjust=False).mean()
    df['smoothed_rs'] = df['rs'] * df['ema_volume_14']
    
    # Calculate Price Momentum Indicator
    df['sma_close_21'] = df['close'].rolling(window=21).mean()
    df['price_momentum'] = df['close'] - df['sma_close_21']
    
    # Incorporate Volume Acceleration
    df['roc_volume_5'] = df['volume'].pct_change(periods=5)
    df['ma_roc_volume_10'] = df['roc_volume_5'].rolling(window=10).mean()
    df['volume_acceleration'] = df['ma_roc_volume_10'] * df['price_momentum']
    
    # Combine Indicators
    df['combined_indicator'] = df['smoothed_rs'] + df['volume_acceleration']
    
    # Cumulative Price-Volume Impulse
    df['price_change'] = df['close'].diff()
    df['adjusted_price_change'] = df.apply(lambda x: x['price_change'] * 2 if x['volume'] > x['volume'].shift(1) else x['price_change'], axis=1)
    df['cumulative_impulse'] = df['adjusted_price_change'].rolling(window=14).sum()
    
    # Final Alpha Factor
    df['alpha_factor'] = df['combined_indicator'] + df['cumulative_impulse']
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
