import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility
    df['volatility'] = df[['high', 'low', 'close']].std(axis=1).rolling(window=30).std()
    
    # Adaptive Window Calculation
    def adjust_window(vol):
        if vol > vol.median():
            return 5  # Decrease window size
        else:
            return 60  # Increase window size
    
    df['window_size'] = df['volatility'].apply(adjust_window)
    
    # Rolling Statistics with Adaptive Window
    df['rolling_mean'] = df.groupby('window_size')['volume_weighted_return'].transform(lambda x: x.rolling(window=x.name).mean())
    df['rolling_std'] = df.groupby('window_size')['volume_weighted_return'].transform(lambda x: x.rolling(window=x.name).std())
    
    # Normalize the factor using rolling mean and standard deviation
    df['factor_value'] = (df['volume_weighted_return'] - df['rolling_mean']) / df['rolling_std']
    
    return df['factor_value']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# factor_values = heuristics_v2(df)
