import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Incorporate Market Microstructure Features (assuming bid and ask are in the dataframe)
    if 'bid' in df.columns and 'ask' in df.columns:
        df['bid_ask_spread'] = df['ask'] - df['bid']
        df['adjusted_volume_weighted_return'] = df['volume_weighted_return'] * df['bid_ask_spread']
    else:
        df['adjusted_volume_weighted_return'] = df['volume_weighted_return']
        print("Bid or Ask not found, skipping bid-ask spread adjustment.")
    
    # Dynamic Volatility Calculation
    df['volatility'] = df[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)
    
    # Adjust Window Size based on Volatility
    volatility_threshold_high = df['volatility'].quantile(0.75)
    volatility_threshold_low = df['volatility'].quantile(0.25)
    
    window_size = np.where(df['volatility'] > volatility_threshold_high, 10,
                           np.where(df['volatility'] < volatility_threshold_low, 30, 20))
    
    # Rolling Statistics with Adaptive Window
    df['rolling_mean'] = [df.loc[i-pd.Timedelta(f'{window_size.iloc[i]}D'):i, 'adjusted_volume_weighted_return'].mean() 
                          for i in df.index]
    df['rolling_std'] = [df.loc[i-pd.Timedelta(f'{window_size.iloc[i]}D'):i, 'adjusted_volume_weighted_return'].std() 
                         for i in df.index]
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['rolling_mean'] / df['rolling_std']).fillna(0)
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# alpha_factor_series = heuristics_v2(df)
