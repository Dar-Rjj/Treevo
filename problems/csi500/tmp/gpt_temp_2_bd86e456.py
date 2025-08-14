import numpy as np
def heuristics_v2(df):
    # Calculate Intraday Return
    df['Intraday_Return'] = (df['high'] - df['low']) / df['open']
    
    # Calculate Overlap Period Return
    df['Overlap_Period_Return'] = (df['close'] - df['open'].shift(1)) / df['open'].shift(1)
    
    # Calculate Volume Growth
    df['Volume_Growth'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Calculate Close Momentum
    n = 5
    df['Close_Momentum'] = sum((df['close'] - df['close'].shift(i)) / df['close'].shift(i) for i in range(1, n))
    
    # Scale High-Low Spread
    df['Scaled_High_Low_Spread'] = (df['high'] - df['low']) * df['close'].rolling(window=n).mean()
    
    # Calculate Volume Weighted Return
    df['Volume_Weighted_Return'] = (df['Intraday_Return'] * df['volume'] + df['Overlap_Period_Return'] * df['volume'].shift(1)) / (df['volume'] + df['volume'].shift(1))
    
    # Integrated Momentum Factor
    df['Integrated_Momentum_Factor'] = (df['Scaled_High_Low_Spread'] * df['Close_Momentum']) + (df['Volume_Weighted_Return'] * df['Volume_Growth'] + df['Overlap_Period_Return'] * (1 - df['Volume_Growth']))
    
    # Calculate Daily High-Low Range
    df['Daily_High_Low_Range'] = df['high'] - df['low']
    
    # Calculate 10-Day Sum of High-Low Ranges
    df['Sum_High_Low_Ranges_10_Days'] = df['Daily_High_Low_Range'].rolling(window=10).sum()
    
    # Calculate Price Change over 10 Days
    df['Price_Change_10_Days'] = df['close'] - df['close'].shift(10)
    
    # Calculate Price Momentum
    df['Price_Trend'] = df['close'] - df['close'].shift(1)
    df['Price_Momentum'] = df['Price_Trend'].ewm(span=21, adjust=False).mean()
    
    # Classify Volume
    lookback_period = 20
    df['Average_Volume'] = df['volume'].rolling(window=lookback_period).mean()
    df['Volume_Classification'] = np.where(df['volume'] > df['Average_Volume'], 'High', 'Low')
    
    # Classify Amount
    df['Average_Amount'] = df['amount'].rolling(window=lookback_period).mean()
    df['Amount_Classification'] = np.where(df['amount'] > df['Average_Amount'], 'High', 'Low')
    
    # Combine Price Momentum with Volume and Amount Classification
    def assign_weights(row):
        if row['Volume_Classification'] == 'High' and row['Amount_Classification'] == 'High':
            return 2.0
