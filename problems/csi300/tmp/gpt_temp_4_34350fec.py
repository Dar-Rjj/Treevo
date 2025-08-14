import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, lookback_period=20, half_life=5, atr_period=14, volume_half_life=5, price_trend_period=30, high_low_period=10):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Compute Cumulative Weighted Return
    weights = np.exp(-np.log(2) / half_life * np.arange(lookback_period))
    weights /= weights.sum()  # Normalize weights
    df['cumulative_weighted_return'] = (df['daily_return'].rolling(window=lookback_period).apply(lambda x: (x * weights).sum(), raw=True))
    
    # Adjust by Volatility
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift()), abs(x['low'] - x['close'].shift())), axis=1)
    df['atr'] = df['true_range'].rolling(window=atr_period).mean()
    df['adjusted_cumulative_return'] = df['cumulative_weighted_return'] / df['atr']
    
    # Incorporate Volume Impact
    df['volume_change'] = df['volume'].pct_change()
    volume_weights = np.exp(-np.log(2) / volume_half_life * np.arange(lookback_period))
    volume_weights /= volume_weights.sum()  # Normalize weights
    df['cumulative_weighted_volume_change'] = (df['volume_change'].rolling(window=lookback_period).apply(lambda x: (x * volume_weights).sum(), raw=True))
    
    # Combine Adjusted Cumulative Weighted Return and Volume Change
    df['combined_factor'] = df['adjusted_cumulative_return'] + df['cumulative_weighted_volume_change']
    
    # Introduce Price Trend Component
    df['moving_avg'] = df['close'].rolling(window=price_trend_period).mean()
    df['price_trend'] = (df['close'] - df['moving_avg']) * (1 / df['close'].std())
    
    # Incorporate High-Low Range
    df['high_low_diff'] = df['high'] - df['low']
    df['avg_high_low_diff'] = df['high_low_diff'].rolling(window=high_low_period).mean()
    df['high_low_range'] = df['high_low_diff'] / df['avg_high_low_diff']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['combined_factor'] + df['price_trend'] + df['high_low_range']
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
