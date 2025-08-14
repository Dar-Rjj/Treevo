import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = df['close'].diff()
    
    # Calculate Price Gap
    df['price_gap'] = df['open'] - df['close'].shift(1)
    
    # Adjust Daily Return by Price Gap
    df['adjusted_daily_return'] = df['daily_return']
    df.loc[df['price_gap'] > 0, 'adjusted_daily_return'] -= df['price_gap'].apply(lambda x: x * np.exp(-1))
    df.loc[df['price_gap'] < 0, 'adjusted_daily_return'] += df['price_gap'].apply(lambda x: x * np.exp(-1))
    
    # Evaluate Momentum Over Time (10-day window)
    window = 10
    df['momentum_10d'] = df['adjusted_daily_return'].rolling(window=window).sum()
    
    # Compute Volume Weighted Momentum
    df['volume_weighted_momentum'] = (df['momentum_10d'] * df['volume']) / df['volume'].rolling(window=window).mean()
    
    # Adjust for Volume Spike Indicator
    df['volume_ma_10d'] = df['volume'].rolling(window=window).mean()
    df['volume_spike'] = df['volume'] > 1.5 * df['volume_ma_10d']
    df['volume_weighted_momentum'] = df['volume_weighted_momentum'] * (1 + 0.2 * df['volume_spike'])
    
    # Adjust for Volatility
    df['historical_volatility'] = df['adjusted_daily_return'].rolling(window=window).std()
    df['volatility_adj'] = (np.abs(df['daily_return']) > 2 * df['historical_volatility']).astype(float) * 0.3
    df['volume_weighted_momentum'] = df['volume_weighted_momentum'] * (1 + df['volatility_adj'])
    
    # Calculate High-to-Low Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Calculate Price Range Momentum
    df['price_range_momentum'] = df['price_range'].diff()
    
    # Adjust by Volume and Price Change
    df['volume_change'] = df['volume'].diff()
    df['price_change'] = df['close'].diff()
    
    # Integrate Momentum, Volume, and Price Change
    df['integrated_momentum'] = df['price_range_momentum'] * df['volume_change'] + df['price_change']
    
    # Combine Adjusted Daily Return with Integrated Momentum
    df['combined_momentum'] = df['adjusted_daily_return'] * df['volume'] + df['integrated_momentum']
    
    # Calculate 5-day Exponential Moving Average for Price
    df['ema_5d_price'] = df['close'].ewm(span=5, adjust=False).mean()
    
    # Calculate Price Momentum
    df['price_momentum'] = df['ema_5d_price'].diff()
    
    # Calculate 5-day Exponential Moving Average for Volume
    df['ema_5d_volume'] = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Calculate Volume Acceleration
    df['volume_acceleration'] = df['ema_5d_volume'].diff()
    
    # Combine Factors
    df['combined_factors'] = (df['volume_acceleration'] * df['price_momentum'] 
                              + df['combined_momentum'] * df['volume_weighted_momentum'])
    
    # Final Alpha Factor with Volume Weighting
    df['final_alpha_factor'] = df['combined_factors'] / df['volume'].rolling(window=5).sum()
    
    # Adjust for Market Sentiment
    df['market_return'] = df['index_close'].diff()
    df['sentiment_multiplier'] = 1.1 * (df['market_return'] > 0) + 0.9 * (df['market_return'] < 0)
    df['final_alpha_factor'] *= df['sentiment_multiplier']
    
    # Ensure Positive Factor
    df['final_alpha_factor'] = df['final_alpha_factor'].apply(lambda x: max(x, 0))
    
    return df['final_alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
