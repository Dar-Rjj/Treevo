import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['vol_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility using High, Low, and Close prices
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['log_returns'] = np.log(df['typical_price']).diff()
    df['volatility'] = df['log_returns'].rolling(window=20).std()
    
    # Adjust Window Size Based on Volatility
    def adaptive_window(volatility):
        if volatility > df['volatility'].quantile(0.75):
            return 10
        elif volatility < df['volatility'].quantile(0.25):
            return 30
        else:
            return 20
    
    df['window_size'] = df['volatility'].apply(adaptive_window)
    
    # Rolling Statistics with Adaptive Window
    df['rolling_mean'] = df['vol_weighted_return'].rolling(window=df['window_size']).mean()
    df['rolling_std'] = df['vol_weighted_return'].rolling(window=df['window_size']).std()
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['vol_weighted_return'] - df['rolling_mean']) / df['rolling_std']
    
    return df['alpha_factor']

# Example usage:
# df = pd.DataFrame(...)  # Load your data into a DataFrame
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
