import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Calculate Smoothed Average True Range (ATR)
    df['atr'] = df['price_range'].rolling(window=14).mean()
    
    # Calculate Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    df['+di'] = 0
    df['-di'] = 0
    for i in range(1, len(df)):
        high_change = df.loc[df.index[i], 'high'] - df.loc[df.index[i-1], 'high']
        low_change = df.loc[df.index[i-1], 'low'] - df.loc[df.index[i], 'low']
        
        df.loc[df.index[i], '+di'] = max(0, high_change)
        df.loc[df.index[i], '-di'] = max(0, low_change)
    
    # Smooth +DI and -DI
    df['+di_smooth'] = df['+di'].rolling(window=14).sum() / df['atr']
    df['-di_smooth'] = df['-di'].rolling(window=14).sum() / df['atr']
    
    # Calculate Directional Movement Index (DX)
    df['dx'] = abs(df['+di_smooth'] - df['-di_smooth']) / (df['+di_smooth'] + df['-di_smooth'])
    
    # Calculate 14-Day Moving Average of DX (ADX)
    df['adx'] = df['dx'].rolling(window=14).mean()
    
    # Weight ADX by Volume
    df['adx_volume_weighted'] = df['adx'] * df['volume']
    
    return df['adx_volume_weighted'].dropna()

# Example usage:
# data = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# factor_values = heuristics_v2(data)
