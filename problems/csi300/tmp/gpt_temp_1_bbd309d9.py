import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_CtoO_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Determine Volatility
    df['True_Range'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    df['Volatility'] = df['True_Range'].rolling(window=20).std()
    
    # Adjust Window Size Based on Volatility
    def adjust_window_size(volatility, low_vol_threshold, high_vol_threshold):
        if volatility < low_vol_threshold:
            return 60  # Increase window size for low volatility
        elif volatility > high_vol_threshold:
            return 5  # Decrease window size for high volatility
        else:
            return 20  # Default window size
    
    low_vol_threshold = 0.01
    high_vol_threshold = 0.03
    df['Adaptive_Window'] = df['Volatility'].apply(adjust_window_size, args=(low_vol_threshold, high_vol_threshold))
    
    # Calculate Rolling Mean and Standard Deviation with Adaptive Window
    def rolling_stats(group):
        window = int(group['Adaptive_Window'].iloc[0])
        mean = group['Volume_Weighted_CtoO_Return'].rolling(window=window).mean()
        std = group['Volume_Weighted_CtoO_Return'].rolling(window=window).std()
        return pd.DataFrame({'Rolling_Mean': mean, 'Rolling_Std': std})
    
    df = df.groupby('Adaptive_Window').apply(rolling_stats).reset_index(level=1, drop=True)
    
    # Final Factor Value
    df['Alpha_Factor'] = (df['Volume_Weighted_CtoO_Return'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    # Drop NA values
    df.dropna(inplace=True)
    
    return df['Alpha_Factor']

# Example usage
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
