import pandas as pd
import pandas as pd

def heuristics_v2(df, m=20, k=14, n=50):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Daily Momentum
    df['daily_momentum'] = df['close'].diff()
    
    # Adjust Momentum by Intraday Range
    df['adjusted_momentum'] = df['daily_momentum'] / df['intraday_range']
    
    # Identify Volume Spikes
    df['avg_volume'] = df['volume'].rolling(window=m).mean()
    df['volume_spike'] = (df['volume'] > df['avg_volume'] * 2).astype(int)
    df['scaled_adjusted_momentum'] = df['adjusted_momentum'] * (1 + (df['volume_spike'] * 0.5))
    
    # Incorporate Trading Activity
    df['true_range'] = df[['high' - 'low', abs('high' - 'close').shift(1), abs('low' - 'close').shift(1)]].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=k).mean()
    df['final_factor'] = df['scaled_adjusted_momentum'] * (1 / df['atr'])
    
    # Combine Adjusted Momentum with Long-Term Price Momentum
    df['long_term_momentum'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n)
    df['final_factor'] = df['final_factor'] * df['long_term_momentum']
    
    return df['final_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
