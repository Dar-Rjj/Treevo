import pandas as pd
import pandas as pd

def heuristics_v2(df, n_days=10, mfi_n_days=14, atr_n_days=14):
    # Momentum Based Indicators
    df['Simple_Price_Movement'] = (df['close'] / df['close'].shift(n_days)) - 1
    df['High_Low_Range_Expansion'] = (df['high'] - df['low']) / df['close']
    
    # Volume Weighted Factors
    df['Volume_Weighted_Price'] = (df['close'] * df['volume']).rolling(window=n_days).sum() / df['volume'].rolling(window=n_days).sum()
    df['Change_in_Volume_Weighted_Price'] = (df['Volume_Weighted_Price'] / df['Volume_Weighted_Price'].shift(1)) - 1
    
    # Money Flow and Strength Indicators
    df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
    df['Raw_Money_Flow'] = df['Typical_Price'] * df['volume']
    df['Positive_Money_Flow'] = df['Raw_Money_Flow'].where(df['Typical_Price'] > df['Typical_Price'].shift(1), 0)
    df['Negative_Money_Flow'] = df['Raw_Money_Flow'].where(df['Typical_Price'] < df['Typical_Price'].shift(1), 0)
    positive_money_flow = df['Positive_Money_Flow'].rolling(window=mfi_n_days).sum()
    negative_money_flow = df['Negative_Money_Flow'].rolling(window=mfi_n_days).sum()
    df['Money_Flow_Index'] = 100 - (100 / (1 + (positive_money_flow / negative_money_flow)))
    
    # Chaikin Oscillator
    df['A_D_Line'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['Chaikin_Oscillator'] = df['A_D_Line'].ewm(span=3).mean() - df['A_D_Line'].ewm(span=10).mean()
    
    # Relative Strength Indicators
    df['True_Range'] = df[['high', 'low', 'close']].join(df['close'].shift(1)).apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    df['Average_True_Range'] = df['True_Range'].rolling(window=atr_n_days).mean()
    
    # Percentage Price Oscillator
    df['EMA_12'] = df['close'].ewm(span=12).mean()
    df['EMA_26'] = df['close'].ewm(span=26).mean()
    df['Percentage_Price_Oscillator'] = ((df['EMA_12'] - df['EMA_26']) / df['EMA_26']) * 100
    
    return df[['Simple_Price_Movement', 'High_Low_Range_Expansion', 'Change_in_Volume_Weighted_Price', 
               'Money_Flow_Index', 'Chaikin_Oscillator', 'Average_True_Range', 'Percentage_Price_Oscillator']]
