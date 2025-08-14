import numpy as np
def heuristics_v2(df):
    # Calculate Intraday Return
    df['Intraday_Return'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Daily Volume Change
    df['Daily_Volume_Change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Calculate Daily Amount Change
    df['Daily_Amount_Change'] = (df['amount'] - df['amount'].shift(1)) / df['amount'].shift(1)
    
    # Calculate Weighted Returns
    df['Weighted_Returns'] = (df['Intraday_Return']**2 + df['Close_to_Open_Return']**2)
    
    # Calculate Adjusted Interim Factor
    df['Adjusted_Interim_Factor'] = df['Weighted_Returns'] * df['Daily_Volume_Change'] * df['Daily_Amount_Change']
    
    # 5-day Moving Average of Adjusted Interim Factor
    df['Interim_Factor_MA5'] = df['Adjusted_Interim_Factor'].rolling(window=5).mean()
    
    # Enhanced Momentum Indicators
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    df['SMA_Cross_Over'] = (df['SMA_7'] > df['SMA_30']).astype(int)
    
    df['EMA_15'] = df['close'].ewm(span=15, adjust=False).mean()
    df['EMA_60'] = df['close'].ewm(span=60, adjust=False).mean()
    df['EMA_Growth'] = df['EMA_15'] - df['EMA_60']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + RS))
    
    # Enhanced Volatility
    df['True_Range'] = df[['high', 'low', 'close']].join(df['close'].shift(1), rsuffix='_Prev').apply(
        lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    df['ATR_20'] = df['True_Range'].rolling(window=20).mean()
    
    df['Volume_Weighted_Price_Change'] = (abs(df['close'] - df['close'].shift(1)) * df['volume'])
    df['Volume_Change_Ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Additional Reversal Indicator
    df['Price_Reversal_10'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['Max_High_20'] = df['high'].rolling(window=20).max()
    df['Min_Low_20'] = df['low'].rolling(window=20).min()
    df['Price_Reversal_20'] = (df['close'] - df['open']) / (df['Max_High_20'] - df['Min_Low_20'])
    
    # Final Composite Alpha Factor
    df['Momentum_Score'] = df['SMA_Cross_Over'] + df['EMA_Growth'] + df['RSI_14']
    df['Volatility_Score'] = df['ATR_20'] + df['Volume_Weighted_Price_Change'] + df['Volume_Change_Ratio']
    df['Reversal_Score'] = df['Price_Reversal_10'] + df['Price_Reversal_20']
    
    df['Composite_Alpha_Factor'] = df['Momentum_Score'] + df['Volatility_Score'] + df['Reversal_Score']
    
    # Sigmoid Function and Scaling to [-1, 1]
    df['Composite_Alpha_Factor'] = 2 * (1 / (1 + np.exp(-df['Composite_Alpha_Factor']))) - 1
    
    return df['Composite
