import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def heuristics(df, n=20):
    # Calculate Daily Price Jerk
    df['Close_Jerk'] = df['close'].diff().diff().diff()
    df['Smoothed_Jerk'] = df['Close_Jerk'].rolling(window=n).sum()

    # Incorporate Volume Adjusted Inertia
    df['Positive_Volume'] = df.apply(lambda row: row['volume'] * (row['close'] - df['close'].shift(1)) if row['close'] > df['close'].shift(1) else 0, axis=1)
    df['Negative_Volume'] = df.apply(lambda row: -row['volume'] * (df['close'].shift(1) - row['close']) if row['close'] <= df['close'].shift(1) else 0, axis=1)
    
    df['Sum_Positive_Volume'] = df['Positive_Volume'].rolling(window=n).sum()
    df['Sum_Negative_Volume'] = df['Negative_Volume'].rolling(window=n).sum().abs()
    
    df['Intermediate_Alpha_Factor'] = df['Smoothed_Jerk'] * (df['Sum_Positive_Volume'] / (df['Sum_Negative_Volume'] + 1e-6))

    # Integrate Enhanced Price-Volume and Spread Dynamics
    df['High_Low_Spread'] = df['high'] - df['low']
    df['Rolling_Pearson_Corr'] = df['close'].rolling(window=n).corr(df['volume'])
    df['Rolling_Spearman_Corr'] = df['High_Low_Spread'].rolling(window=n).apply(lambda x: spearmanr(x, df.loc[x.index, 'volume'])[0], raw=False)
    
    df['MA_High_Low_Spread'] = df['High_Low_Spread'].rolling(window=n).mean()
    df['Relative_High_Low_Spread'] = df['High_Low_Spread'] / (df['MA_High_Low_Spread'] + 1e-6)

    df['Final_Alpha_Factor'] = (df['Intermediate_Alpha_Factor'] * 
                                df['Rolling_Pearson_Corr'] * 
                                df['Rolling_Spearman_Corr'] * 
                                df['Relative_High_Low_Spread'])

    return df['Final_Alpha_Factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics(df)
