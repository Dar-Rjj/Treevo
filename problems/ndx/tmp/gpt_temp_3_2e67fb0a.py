import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Daily Momentum Contribution
    df['Price_Change'] = df['close'] - df['open'].shift(1)
    df['Volume_Impact'] = np.sqrt(df['volume']) * abs(df['Price_Change'])
    
    # Historical Momentum Integration
    df['Momentum_Contribution'] = df['Price_Change'] * df['Volume_Impact']
    df['Sum_Last_7_Days_Momentum'] = df['Momentum_Contribution'].rolling(window=7).sum()
    df['Volume_Weighted_Avg'] = (df['Momentum_Contribution'].rolling(window=7).apply(lambda x: (np.sqrt(x) * df.loc[x.index, 'Momentum_Contribution']).sum() / np.sqrt(x).sum(), raw=False))
    
    # Market Sentiment and Price Oscillation Adjustment
    df['Volatility_Threshold'] = (df['high'] - df['low']) / df['close']
    df['Avg_Volatility_Threshold_7d'] = df['Volatility_Threshold'].rolling(window=7).mean()
    
    # Trend Confirmation
    df['Trend_Confirmation'] = np.where(df['open'] > df['close'].shift(1), 1, -1)
    
    # 5-Day Price Range Adjustment
    df['5_Day_High'] = df['high'].rolling(window=5).max()
    df['5_Day_Low'] = df['low'].rolling(window=5).min()
    df['5_Day_Range'] = (df['5_Day_High'] - df['5_Day_Low']) / df['close']
    df['5_Day_Average_Range'] = df['5_Day_Range'].rolling(window=5).mean()
    df['Range_Adjustment_Multiplier'] = np.where(df['5_Day_Range'] > df['5_Day_Average_Range'], 1.2, 0.8)
    
    # Final EVPOI Calculation
    df['EVPOI'] = df['Volume_Weighted_Avg'] * df['Trend_Confirmation'] * df['Range_Adjustment_Multiplier']
    
    return df['EVPOI']
