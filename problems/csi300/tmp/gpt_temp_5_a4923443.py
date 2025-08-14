import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the percentage change in close price
    df['pct_change_close'] = df['close'].pct_change()
    
    # Calculate the average of high and low
    df['avg_high_low'] = (df['high'] + df['low']) / 2
    
    # Calculate the ratio of volume to amount
    df['volume_to_amount_ratio'] = df['volume'] / df['amount']
    
    # Calculate the weighted moving average of the close price using the volume as weights
    df['wma_close'] = df['close'].rolling(window=10).apply(lambda x: (x * df.loc[x.index, 'volume']).sum() / df.loc[x.index, 'volume'].sum(), raw=False)
    
    # Heuristic factor: A combination of the above calculated metrics
    df['heuristic_factor'] = df['pct_change_close'] + df['avg_high_low'] * df['volume_to_amount_ratio'] - df['wma_close']
    
    # Return the heuristic factor as a pandas Series
    return df['heuristic_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
# print(factor_values)
