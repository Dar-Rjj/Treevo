import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['Close_t'] = df['close']
    df['Open_t1'] = df['open'].shift(-1)
    df['Simple_Returns'] = (df['Open_t1'] - df['Close_t']) / df['Close_t']
    df['Volume_Weighted_Returns'] = df['Simple_Returns'] * df['volume']

    # Identify Volume Surge Days
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Volume_Rolling_Mean'] = df['volume'].rolling(window=5).mean()
    df['Is_Volume_Surge'] = df['volume'] > df['Volume_Rolling_Mean']

    # Calculate Adaptive Volatility
    df['Daily_Returns'] = df['close'].pct_change()
    def adjust_lookback(std, recent_vol):
        if recent_vol > 0.01:
            return 20
        elif recent_vol > 0.005:
            return 40
        else:
            return 60
    df['Volatility'] = df['Daily_Returns'].rolling(window=lambda x: adjust_lookback(x, df['Daily_Returns'].std())).std()
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    df['Volume_Std'] = df['volume'].rolling(window=20).std()
    df['Volume_Z_Score'] = (df['volume'] - df['Volume_MA']) / df['Volume_Std']
    df['Adaptive_Volatility'] = df['Volatility'] * (1 + np.abs(df['Volume_Z_Score']))

    # Refine Volume Surge Factors
    df['Volume_Surge_Ratio'] = df['volume'] / df['volume'].shift(1)
    df['Surge_Factor'] = 1
    df.loc[df['Volume_Surge_Ratio'] > 2.5, 'Surge_Factor'] = 1.8
    df.loc[(df['Volume_Surge_Ratio'] > 2.0) & (df['Volume_Surge_Ratio'] <= 2.5), 'Surge_Factor'] = 1.5
    df.loc[(df['Volume_Surge_Ratio'] > 1.5) & (df['Volume_Surge_Ratio'] <= 2.0), 'Surge_Factor'] = 1.2

    # Adjust Volume-Weighted Returns by Adaptive Volatility
    df['Adjusted_Volume_Weighted_Returns'] = df['Volume_Weighted_Returns'] / df['Adaptive_Volatility']

    # Combine Adjusted Returns with Refined Volume Surge Indicator
    df['Factor'] = df['Adjusted_Volume_Weighted_Returns']
    df.loc[df['Is_Volume_Surge'], 'Factor'] *= df['Surge_Factor']

    return df['Factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
# print(factor_values)
