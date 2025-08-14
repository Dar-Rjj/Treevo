import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Price Momentum
    df['price_momentum'] = df['close'].diff()
    
    # Calculate Volume Adjusted Momentum
    average_volume = df['volume'].mean()
    df['volume_adjusted_momentum'] = (df['close'].diff() * (df['volume'] / average_volume))
    
    # Calculate True Range for Volatility
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], 
                      abs(x['high'] - df['close'].shift(1)), 
                      abs(x['low'] - df['close'].shift(1))), axis=1)
    
    # Normalize Momentum by Volatility
    df['normalized_momentum'] = df['volume_adjusted_momentum'] / df['true_range']
    
    return df['normalized_momentum'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(df)
