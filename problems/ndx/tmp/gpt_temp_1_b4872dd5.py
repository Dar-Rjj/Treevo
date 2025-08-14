import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10, m=14):
    # Calculate Daily Return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Volume Change Ratio
    df['volume_change_ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Compute Weighted Momentum
    df['weighted_momentum'] = (df['daily_return'] * df['volume_change_ratio']).rolling(window=n).sum()
    
    # Calculate True Range
    df['true_range'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    
    # Calculate ATR (Average True Range) over m days
    df['atr'] = df['true_range'].rolling(window=m).mean()
    
    # Adjust for Price Volatility
    df['adjusted_momentum'] = df['weighted_momentum'] - df['atr']
    
    # Return the factor values
    return df['adjusted_momentum']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
