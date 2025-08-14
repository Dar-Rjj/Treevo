import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']

    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']

    # Comprehensive Volatility Measure
    df['True_Range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['Avg_True_Range'] = df['True_Range'].rolling(window=20).mean()
    df['Std_Close'] = df['close'].rolling(window=20).std()

    # Determine Adaptive Window Size
    adaptive_window = 30  # Initial window size
    df['Adaptive_Window_Size'] = adaptive_window + (df['Avg_True_Range'].diff() / df['Avg_True_Range'].diff().mean()) * 5
    df['Adaptive_Window_Size'] = df['Adaptive_Window_Size'].clip(lower=10, upper=60)

    # Rolling Mean and Standard Deviation of Volume Weighted Close-to-Open Return
    df['Rolling_Mean_Volume_Weighted_Return'] = df.groupby('Volume_Weighted_Return').rolling(window=df['Adaptive_Window_Size']).mean().reset_index(0, drop=True)
    df['Rolling_Std_Volume_Weighted_Return'] = df.groupby('Volume_Weighted_Return').rolling(window=df['Adaptive_Window_Size']).std().reset_index(0, drop=True)

    # Adjust for Volatility
    df['Comprehensive_Volatility'] = (df['Avg_True_Range'] + df['Std_Close']) / 2
    df['Volatility_Adjusted_Return'] = df['Volume_Weighted_Return'] / df['Comprehensive_Volatility']

    # Incorporate Intraday Data
    df['Intraday_Range'] = df['high'] - df['low']
    df['Adjusted_Alpha_Factor'] = df['Volatility_Adjusted_Return'] * df['Intraday_Range']

    # Incorporate Price Momentum
    df['MA_5_Day'] = df['close'].rolling(window=5).mean()
    df['Momentum_5_Day'] = df['MA_5_Day'] - df['close']

    # Combine Factors
    df['Alpha_Factor'] = df['Adjusted_Alpha_Factor'] + df['Momentum_5_Day']

    return df['Alpha_Factor']

# Example usage
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
