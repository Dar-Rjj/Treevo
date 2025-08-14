import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute 3-Day Log Return
    df['3_day_log_return'] = np.log(df['close']).diff(3)
    
    # Compute 5-Day Log Return
    df['5_day_log_return'] = np.log(df['close']).diff(5)
    
    # Compute 10-Day Log Return
    df['10_day_log_return'] = np.log(df['close']).diff(10)
    
    # Weighted Average of 3-Day, 5-Day, and 10-Day Returns
    df['weighted_log_return'] = (df['3_day_log_return'] * 0.3 + 
                                 df['5_day_log_return'] * 0.4 + 
                                 df['10_day_log_return'] * 0.3)
    
    # Multiply by Volume of day t
    df['volume_adjusted_momentum'] = df['weighted_log_return'] * df['volume']
    
    # Compute High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']
    
    # Compute Open-Close Spread
    df['open_close_spread'] = abs(df['open'] - df['close'])
    
    # Combine Spreads
    df['combined_spread'] = (df['high_low_spread'] + df['open_close_spread']) / 2
    
    # Compute Mean Price
    df['mean_price'] = (df['open'] + df['close']) / 2
    
    # Normalize Combined Spread by Mean Price
    df['normalized_spread'] = df['combined_spread'] / df['mean_price']
    
    # Compute 10-Day Average True Range (ATR)
    def true_range(row):
        return max(row['high'] - row['low'], 
                   abs(row['high'] - row['close'].shift(1)), 
                   abs(row['low'] - row['close'].shift(1)))
    
    df['true_range'] = df.apply(true_range, axis=1)
    df['10_day_atr'] = df['true_range'].rolling(window=10).mean()
    
    # Adjust for Volatility
    df['enhanced_volatility_index'] = df['normalized_spread'] / df['10_day_atr']
    
    # Final Factor Calculation
    df['alpha_factor'] = df['volume_adjusted_momentum'] - df['enhanced_volatility_index']
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
