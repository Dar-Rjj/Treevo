import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Dynamic Volatility Adjustment
    def calculate_volatility(prices, window=20):
        return prices.rolling(window=window).std()
    
    df['Volatility'] = calculate_volatility(df[['high', 'low', 'close']].mean(axis=1))
    
    volatility_median = df['Volatility'].median()
    df['Adjusted_Volume_Weighted_Return'] = df['Volume_Weighted_Return'] * (volatility_median / df['Volatility'])
    
    # Adaptive Window Calculation
    initial_window = 20
    df['Adaptive_Window'] = np.where(df['Volatility'] > volatility_median, 
                                     initial_window * 0.5, 
                                     initial_window * 1.5)
    df['Adaptive_Window'] = df['Adaptive_Window'].astype(int)
    
    # Rolling Statistics with Adaptive Window
    def rolling_stats(series, window_series):
        return series.rolling(window=window_series, min_periods=initial_window).agg(['mean', 'std'])
    
    result = df.groupby('Adaptive_Window').apply(lambda x: rolling_stats(x['Adjusted_Volume_Weighted_Return'], x['Adaptive_Window']))
    
    # Unstack the MultiIndex to get a DataFrame with mean and std as columns
    result = result.unstack().swaplevel(axis=1).sort_index(axis=1)
    
    # Return the final alpha factor
    return result['mean'].droplevel(0)

# Example usage:
# data = pd.read_csv('your_data.csv')
# data.set_index('date', inplace=True)
# alpha_factor = heuristics_v2(data)
# print(alpha_factor)
