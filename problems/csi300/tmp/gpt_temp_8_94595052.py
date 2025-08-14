import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['Open_t1'] = df['open'].shift(-1)
    df['Simple_Returns'] = (df['Open_t1'] - df['close']) / df['close']
    df['Volume_Weighted_Returns'] = df['Simple_Returns'] * df['volume']

    # Identify Volume Surge Days
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Rolling_Volume_Mean'] = df['volume'].rolling(window=5).mean()
    df['Volume_Surge'] = df['volume'] > df['Rolling_Volume_Mean']
    
    # Calculate Volatility Using ATR
    df['Prev_Close'] = df['close'].shift(1)
    df['True_Range'] = df[['high' - 'low', abs('high' - 'Prev_Close'), abs('low' - 'Prev_Close')]].max(axis=1)
    N_atr = 14  # Adaptive based on market conditions
    df['ATR'] = df['True_Range'].rolling(window=N_atr).mean()

    # Adjust Volume-Weighted Returns by ATR
    df['Adjusted_Returns'] = df['Volume_Weighted_Returns'] / df['ATR']

    # Calculate Relative Strength Index (RSI)
    df['Price_Change'] = df['close'] - df['close'].shift(1)
    df['Gain'] = df['Price_Change'].apply(lambda x: x if x > 0 else 0)
    df['Loss'] = abs(df['Price_Change'].apply(lambda x: x if x < 0 else 0))
    N_rsi = 14  # Adaptive based on market conditions
    df['Avg_Gain'] = df['Gain'].rolling(window=N_rsi).mean()
    df['Avg_Loss'] = df['Loss'].rolling(window=N_rsi).mean()
    df['RS'] = df['Avg_Gain'] / df['Avg_Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))

    # Combine Adjusted Returns with Volume Surge Indicator and RSI
    surge_factor = 1.5
    df['Combined_Returns'] = df['Adjusted_Returns']
    df.loc[df['Volume_Surge'], 'Combined_Returns'] *= surge_factor
    
    # Adjust Combined Returns by RSI
    df['Final_Alpha_Factor'] = df['Combined_Returns']
    df.loc[df['RSI'] < 30, 'Final_Alpha_Factor'] *= 1.2
    df.loc[df['RSI'] > 70, 'Final_Alpha_Factor'] *= 0.8

    return df['Final_Alpha_Factor'].dropna()

# Example usage:
# alpha_factor = heuristics_v2(your_dataframe)
