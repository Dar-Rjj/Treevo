import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Determine Volatility (Using Garman-Klass estimator for simplicity)
    df['Log_High_Low'] = 0.5 * (np.log(df['high']) - np.log(df['low']))**2
    df['Log_Close_Open'] = (np.log(df['close']) - np.log(df['open']))**2
    df['Volatility'] = (df['Log_High_Low'] + df['Log_Close_Open']).rolling(window=30).mean() ** 0.5
    
    # Adaptive Window Calculation
    def adaptive_window(volatility, low_vol_threshold=0.001, high_vol_threshold=0.01, base_window=60):
        if volatility < low_vol_threshold:
            return base_window * 2  # Increase window size
        elif volatility > high_vol_threshold:
            return max(1, int(base_window / 2))  # Decrease window size
        else:
            return base_window  # Keep the base window size
    
    # Apply adaptive window to each row
    df['Adaptive_Window'] = df['Volatility'].apply(adaptive_window)
    
    # Rolling Mean and Standard Deviation with Adaptive Windows
    df['Rolling_Mean'] = df.groupby('date')['Volume_Weighted_Return'].transform(lambda x: x.rolling(window=df.loc[x.index, 'Adaptive_Window'][0]).mean())
    df['Rolling_Std'] = df.groupby('date')['Volume_Weighted_Return'].transform(lambda x: x.rolling(window=df.loc[x.index, 'Adaptive_Window'][0]).std())
    
    # Final Factor: Standardized Volume-Weighted Return
    df['Factor'] = (df['Volume_Weighted_Return'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    return df['Factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
