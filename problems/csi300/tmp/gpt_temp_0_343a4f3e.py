import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Adaptive Window Calculation
    def calculate_volatility(high, low, close, window=20):
        avg_price = (high + low + close) / 3
        return avg_price.rolling(window=window).std()
    
    initial_window = 20
    df['volatility'] = calculate_volatility(df['high'], df['low'], df['close'], window=initial_window)
    
    volatility_avg = df['volatility'].mean()
    df['adaptive_window'] = initial_window * (1 + (df['volatility'] - volatility_avg) / volatility_avg)
    df['adaptive_window'] = df['adaptive_window'].apply(lambda x: max(5, min(40, int(x))))  # Bound the window size
    
    # Rolling Statistics
    df['rolling_mean'] = df.groupby('volume_weighted_return').rolling(window=df['adaptive_window']).mean().reset_index(0, drop=True)
    df['rolling_std'] = df.groupby('volume_weighted_return').rolling(window=df['adaptive_window']).std().reset_index(0, drop=True)
    
    # Additional Price Features
    df['high_low_spread'] = df['high'] - df['low']
    df['high_close_spread'] = df['high'] - df['close']
    
    # Dynamic Volatility Adjustment
    df['close_rolling_std'] = df['close'].rolling(window=initial_window).std()
    df['volatility_ratio'] = df['close_rolling_std'] / df['close_rolling_std'].mean()
    df['adjusted_close_to_open_return'] = df['close_to_open_return'] * (1 / (1 + df['volatility_ratio']))
    df['adjusted_volume_weighted_return'] = df['adjusted_close_to_open_return'] * df['volume']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = (df['rolling_mean'] / df['rolling_std']) * df['adjusted_volume_weighted_return'] + \
                               df['high_low_spread'] + df['high_close_spread']
    
    return df['final_alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
