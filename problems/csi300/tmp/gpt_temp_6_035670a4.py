import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, unemployment_rate, inflation_rate):
    # Calculate Logarithmic Returns
    df['log_returns'] = np.log(df['close']).diff()
    
    # Calculate Historical Volatility (20 Days)
    df['volatility'] = df['close'].rolling(window=20).std()
    avg_volatility = df['volatility'].mean()
    
    # Adjust Window Based on Recent Volatility
    def adjust_window(row):
        if row['volatility'] > 1.5 * avg_volatility:
            return 10
        elif row['volatility'] < 0.5 * avg_volatility:
            return 30
        else:
            return 20
    
    df['momentum_window'] = df.apply(adjust_window, axis=1)
    df['price_momentum'] = df['log_returns'].rolling(window=df['momentum_window']).sum()
    
    # Calculate Volume Acceleration
    df['volume_change'] = df['volume'].pct_change()
    df['volume_variability'] = df['volume'].rolling(window=10).std()
    avg_variability = df['volume_variability'].mean()
    
    # Adjust Window Based on Recent Volume Variability
    def adjust_volume_window(row):
        if row['volume_variability'] > 1.5 * avg_variability:
            return 5
        elif row['volume_variability'] < 0.5 * avg_variability:
            return 20
        else:
            return 10
    
    df['acceleration_window'] = df.apply(adjust_volume_window, axis=1)
    df['volume_acceleration'] = df['volume_change'].rolling(window=df['acceleration_window']).sum()
    
    # Incorporate Volatility
    def adjust_factor(row):
        if row['volatility'] > 1.5 * avg_volatility:
            return row['price_momentum'] * 0.8
        else:
            return row['price_momentum'] * 1.2
    
    df['adjusted_momentum'] = df.apply(adjust_factor, axis=1)
    
    # Integrate Macroeconomic Indicators
    macro_adjustment = 1.0
    if unemployment_rate > 5.0:
        macro_adjustment *= 0.9
    if inflation_rate > 2.0:
        macro_adjustment *= 1.1
    
    # Combine Momentum and Acceleration with Adjustments
    df['alpha_factor'] = (df['adjusted_momentum'] * df['volume_acceleration']) * macro_adjustment
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df, unemployment_rate=4.5, inflation_rate=2.5)
# print(alpha_factor)
