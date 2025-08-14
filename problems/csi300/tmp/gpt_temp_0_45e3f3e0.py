import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, window=5, alpha=0.9):
    """
    Calculate the integrated and transformed intraday momentum and volatility.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing (date, open, high, low, close, amount, volume)
    window (int): Rolling window size for momentum calculation
    alpha (float): Smoothing factor for exponential smoothing
    
    Returns:
    pd.Series: Factor values indexed by date
    """
    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Weight by Volume
    df['weighted_volatility'] = df['intraday_volatility'] * df['volume']
    
    # Calculate Intraday Momentum
    df['daily_momentum'] = df['close'] - df['open']
    df['rolling_momentum'] = df['daily_momenturm'].rolling(window=window).sum()
    
    # Integrate Momentum and Volatility
    df['integrated'] = df['weighted_volatility'] + df['rolling_momentum']
    
    # Apply Dynamic Exponential Smoothing
    df['smoothed'] = df['integrated'].ewm(alpha=alpha).mean()
    
    # Ensure Values are Positive
    df['positive_smoothed'] = df['smoothed'] + 1e-6
    
    # Apply Logarithmic Transformation
    df['log_transformed'] = np.log(df['positive_smoothed'])
    
    # Return the final factor values
    return df['log_transformed']

# Example usage
# df = pd.DataFrame({
#     'date': pd.date_range(start='2023-01-01', periods=20),
#     'open': np.random.rand(20) * 100,
#     'high': np.random.rand(20) * 100,
#     'low': np.random.rand(20) * 100,
#     'close': np.random.rand(20) * 100,
#     'amount': np.random.rand(20) * 1000,
#     'volume': np.random.randint(100, 1000, 20)
# })
# df.set_index('date', inplace=True)
# factor_values = heuristics_v2(df)
# print(factor_values)
