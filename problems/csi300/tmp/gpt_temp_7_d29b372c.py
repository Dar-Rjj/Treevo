import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']

    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']

    # Determine Volatility using High, Low, and Close prices
    df['Volatility'] = df[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)

    # Adjust Window Size based on Volatility
    def adaptive_window(vol):
        if vol > vol.median():
            return 10  # Decrease window size for high volatility
        else:
            return 30  # Increase window size for low volatility

    df['Adaptive_Window'] = df['Volatility'].apply(adaptive_window)

    # Rolling Statistics with Adaptive Window
    def rolling_stat(x):
        window = int(x['Adaptive_Window'].iloc[0])
        mean = x['Volume_Weighted_Return'].rolling(window=window).mean().iloc[-1]
        std = x['Volume_Weighted_Return'].rolling(window=window).std().iloc[-1]
        return pd.Series([mean, std], index=['Rolling_Mean', 'Rolling_Std'])

    grouped = df.groupby(df.index.date)
    rolling_stats = grouped.apply(rolling_stat)
    df = df.join(rolling_stats, on=df.index.date)

    # Final Factor: Standardized Rolling Mean of Volume Weighted Close-to-Open Return
    df['Alpha_Factor'] = (df['Rolling_Mean'] - df['Rolling_Mean'].mean()) / df['Rolling_Mean'].std()

    return df['Alpha_Factor']
