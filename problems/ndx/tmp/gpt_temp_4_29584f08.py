import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Daily Momentum Contribution
    df['Price_Change'] = df['close'] - df['open'].shift(1)
    df['Volume_Impact'] = np.sqrt(df['volume']) * abs(df['Price_Change'])
    df['Momentum_Contribution'] = df['Price_Change'] + df['Volume_Impact']
    
    # Historical Momentum Integration
    df['Sum_Momentum_Last_7'] = df['Momentum_Contribution'].rolling(window=7).sum()
    df['Weighted_Average_Volume'] = (df['Volume_Impact'].rolling(window=7).sum()) / (np.sqrt(df['volume']).rolling(window=7).sum())
    
    # Market Sentiment and Price Oscillation Adjustment
    df['Volatility_Threshold'] = (df['high'] - df['low']) / df['close']
    df['Avg_Volatility_Threshold_7'] = df['Volatility_Threshold'].rolling(window=7).mean()
    
    # Trend Confirmation
    df['EVPOI_Adjustment'] = df['Open'] > df['close'].shift(1)
    df['EVPOI_Adjustment'] = df['EVPOI_Adjustment'].apply(lambda x: 1.1 if x else 0.9)
    
    # 5-Day Price Range Adjustment
    df['5_Day_High'] = df['high'].rolling(window=5).max()
    df['5_Day_Low'] = df['low'].rolling(window=5).min()
    df['5_Day_Range'] = (df['5_Day_High'] - df['5_Day_Low']) / df['close']
    df['5_Day_Avg_Range'] = df['5_Day_Range'].rolling(window=5).mean()
    df['Range_Multiplier'] = df['5_Day_Range'] > df['5_Day_Avg_Range']
    df['Range_Multiplier'] = df['Range_Multiplier'].apply(lambda x: 1.2 if x else 0.8)
    
    # 3-Day Price Momentum Smoothing
    df['3_Day_Price_Momentum'] = (df['close'] - df['close'].shift(3)) / 3
    df['Smoothing_Factor'] = df['3_Day_Price_Momentum'].apply(lambda x: 1.1 if x > 0 else 0.9)
    
    # Final EVPOI Calculation
    df['EVPOI'] = (df['Sum_Momentum_Last_7'] * df['Weighted_Average_Volume'] * df['EVPOI_Adjustment'] * df['Range_Multiplier'] * df['Smoothing_Factor'])
    
    return df['EVPOI'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# evpoi = heuristics_v2(df)
# print(evpoi)
