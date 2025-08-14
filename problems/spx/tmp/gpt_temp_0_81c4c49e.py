import pandas as pd
import pandas as pd

def heuristics_v2(df, n=5):
    # Calculate Price Momentum
    df['Close_t1'] = df['close'].shift(1)
    df['Close_tn'] = df['close'].shift(n)
    df['Price_Momentum'] = (df['Close_t1'] - df['Close_tn']) / df['Close_tn']
    
    # Compute Weighted Average Volume and Amount
    df['Volume_Sum'] = df['volume'].rolling(window=n).sum()
    df['Amount_Sum'] = df['amount'].rolling(window=n).sum()
    df['Weighted_Average_Volume'] = df['Volume_Sum'] / n
    df['Weighted_Average_Amount'] = df['Amount_Sum'] / n
    
    # Adjust Volume and Amount
    df['Adjusted_Volume'] = df['volume'] - df['Weighted_Average_Volume']
    df['Adjusted_Amount'] = df['amount'] - df['Weighted_Amount_Sum']
    
    # Multiply Price Momentum by Adjusted Volume and Amount
    df['Momentum_Volume_Adjusted'] = df['Price_Momentum'] * df['Adjusted_Volume']
    df['Momentum_Amount_Adjusted'] = df['Price_Momentum'] * df['Adjusted_Amount']
    df['Factor_Value'] = df['Momentum_Volume_Adjusted'] + df['Momentum_Amount_Adjusted']
    
    return df['Factor_Value'].dropna()
