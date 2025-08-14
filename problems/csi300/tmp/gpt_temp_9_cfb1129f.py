import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Calculate Close-to-Open Return
    data['Close_to_Open_Return'] = data['open'].shift(-1) - data['close']

    # Volume Weighting
    data['Volume_Weighted_Return'] = data['Close_to_Open_Return'] * data['volume']

    # Determine Volatility
    data['Volatility'] = data[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)

    # Incorporate Trading Patterns
    data['Average_Amount'] = data['amount'].rolling(window=20).mean()

    # Adjust Window Size
    def adjust_window_size(vol, amount):
        if vol > 0.5 * data['Volatility'].median() and amount < 0.5 * data['amount'].median():
            return 10  # Decrease window size
        elif vol < 0.5 * data['Volatility'].median() and amount > 1.5 * data['amount'].median():
            return 30  # Increase window size
        else:
            return 20  # Default window size

    data['Adaptive_Window'] = data.apply(lambda row: adjust_window_size(row['Volatility'], row['Average_Amount']), axis=1)
    # To apply the adaptive window, we need to use a loop or more advanced techniques which are not straightforward in Pandas.
    # Here we simplify by using the median of the adaptive windows for the rolling statistics.
    median_window = data['Adaptive_Window'].median()
    
    # Rolling Statistics with Adaptive Window
    data['Rolling_Mean'] = data['Volume_Weighted_Return'].rolling(window=int(median_window)).mean()
    data['Rolling_Std'] = data['Volume_Weighted_Return'].rolling(window=int(median_window)).std()

    # The final factor is a volatility-adjusted rolling mean
    data['Final_Factor'] = data['Rolling_Mean'] / (data['Rolling_Std'] + 1e-6)  # Adding small constant to avoid division by zero

    return data['Final_Factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# result = heuristics_v2(df)
# print(result)
