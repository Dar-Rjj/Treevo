import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the price change from the previous day
    df['price_change'] = df['close'].diff()
    
    # Calculate the average price
    df['avg_price'] = (df['high'] + df['low']) / 2
    
    # Calculate the money flow by multiplying the average price with the volume
    df['money_flow'] = df['avg_price'] * df['volume']
    
    # Calculate the typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate the true range
    df['true_range'] = df[['high' - 'low', 
                           ('high' - 'close').abs(), 
                           ('low' - 'close').abs()]].max(axis=1)
    
    # Calculate the alpha factor
    df['alpha_factor'] = (df['money_flow'] / df['amount']) * (df['price_change'] / df['true_range'])
    
    # Return the alpha factor as a pandas Series
    return df['alpha_factor']

# Example usage:
# df = pd.DataFrame({
#     'open': [...],
#     'high': [...],
#     'low': [...],
#     'close': [...],
#     'amount': [...],
#     'volume': [...]
# }, index=pd.to_datetime([...]))
# alpha_factor = heuristics_v2(df)
