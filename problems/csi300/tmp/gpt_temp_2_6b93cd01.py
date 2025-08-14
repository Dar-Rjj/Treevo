import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=5, C=100):
    # Calculate momentum based on close prices
    df['Delta'] = df['close'].diff(1)
    
    # Incorporate volume into momentum
    df['Scaled_Momentum'] = (df['Delta'] / np.sqrt(df['volume'])) * C
    
    # Initialize or retrieve from memory the rolling window for past N days
    if 'Past_Scaled_Momenta' not in df.columns:
        df['Past_Scaled_Momenta'] = df.apply(lambda row: [row['Scaled_Momentum']], axis=1)
    else:
        df['Past_Scaled_Momenta'] = df.apply(
            lambda row: (df.loc[row.name - pd.Timedelta(days=1)]['Past_Scaled_Momenta'] + [row['Scaled_Momentum']])[-N:], 
            axis=1
        )
    
    # Assign weights (e.g., exponentially decreasing)
    weights = np.exp(-np.arange(N) / N)
    weights /= weights.sum()  # Normalize weights to sum to 1
    
    # Weighted sum of the past N scaled momenta
    df['Dynamic_Momentum_Indicator'] = df['Past_Scaled_Momenta'].apply(
        lambda x: np.dot(x, weights)
    )
    
    # Calculate standard deviation of close prices over the past N days
    df['Close_STD'] = df['close'].rolling(window=N).std()
    
    # Adjust scaled momentum by volatility
    df['Adjusted_Scaled_Momentum'] = df['Dynamic_Momentum_Indicator'] / df['Close_STD']
    
    return df['Adjusted_Scaled_Momentum']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# alpha_factor = heuristics_v2(df)
