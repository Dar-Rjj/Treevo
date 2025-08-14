import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10, m=14):
    # Calculate Daily Return
    df['Daily_Return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Volume Change Ratio
    df['Volume_Change_Ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Compute Weighted Momentum
    df['Weighted_Momentum'] = (df['Daily_Return'] * df['Volume_Change_Ratio']).rolling(window=n).sum()
    
    # True Range
    df['True_Range'] = df.apply(lambda row: max(row['high'] - row['low'], 
                                                abs(row['high'] - row['close'].shift(1)), 
                                                abs(row['low'] - row['close'].shift(1))), axis=1)
    
    # Average True Range (ATR) over m days
    df['ATR'] = df['True_Range'].rolling(window=m).mean()
    
    # Enhanced ATR
    df['Enhanced_ATR'] = df['ATR'] * (1 + 0.5 * (df['high'] - df['low']) / df['close'].shift(1))
    
    # Final Factor
    df['Final_Factor'] = df['Weighted_Momentum'] - df['Enhanced_ATR']
    
    return df['Final_Factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor = heuristics_v2(df)
# print(factor)
