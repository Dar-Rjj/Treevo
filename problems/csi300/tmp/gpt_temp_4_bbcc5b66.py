import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    df['Intraday_High_Low_Spread'] = df['High'] - df['Low']
    
    # Compute Previous Day's Open-to-Close Return
    df['Prev_Open'] = df['Open'].shift(1)
    df['Open_Close_Return'] = df['Close'] - df['Prev_Open']
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['Daily_VWAP'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Combine Intraday Momentum and VWAP
    df['Combined_Value'] = df['Daily_VWAP'] - df['Intraday_High_Low_Spread']
    df['Weighted_Combined_Value'] = df['Combined_Value'] * df['Volume']
    
    # Incorporate Volatility
    df['7_Day_Volatility'] = df['Close'].rolling(window=7).std()
    
    # Adaptive Smoothing
    def adaptive_ema(span, series):
        if span == 5:
            return series.ewm(span=5).mean()
        elif span == 10:
            return series.ewm(span=10).mean()
    
    df['Smoothed_Factor'] = df.apply(lambda row: adaptive_ema(5, row['Weighted_Combined_Value']) if row['7_Day_Volatility'] > df['7_Day_Volatility'].median() else adaptive_ema(10, row['Weighted_Combined_Value']), axis=1)
    
    # Consider Liquidity
    df['ADV'] = df['Volume'].rolling(window=20).mean()
    mean_ADV_60 = df['Volume'].rolling(window=60).mean().mean()
    
    def damping_factor(adv, mean_adv_60):
        if adv < 0.5 * mean_adv_60:
            return 0.8
        elif adv > 1.5 * mean_adv_60:
            return 0.9
        else:
            return 1.0
    
    df['Damping_Factor'] = df['ADV'].apply(lambda x: damping_factor(x, mean_ADV_60))
    
    # Final Factor
    df['Final_Factor'] = df['Smoothed_Factor'] * df['Damping_Factor']
    
    return df['Final_Factor']

# Example usage:
# df = pd.DataFrame({
#     'Date': pd.date_range(start='2023-01-01', periods=100),
#     'Open': np.random.rand(100) * 100,
#     'High': np.random.rand(100) * 100,
#     'Low': np.random.rand(100) * 100,
#     'Close': np.random.rand(100) * 100,
#     'Amount': np.random.rand(100) * 10000,
#     'Volume': np.random.randint(1000, 5000, size=100)
# })
# df.set_index('Date', inplace=True)
# factor_values = heuristics_v2(df)
# print(factor_values)
