import numpy as np
def heuristics_v2(df):
    # Calculate Daily Price Range
    df['Daily_Price_Range'] = df['High'] - df['Low']
    
    # Calculate Average Price
    df['Average_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Compute Volume Adjusted Range
    df['Volume_Adjusted_Range'] = (df['Daily_Price_Range'] / df['Average_Price']).replace([np.inf, -np.inf], 0) * df['Volume']
    
    # Calculate Short-Term and Long-Term Rolling Averages
    df['Short_Term_Rolling_Avg'] = df['Volume_Adjusted_Range'].rolling(window=10).mean()
    df['Long_Term_Rolling_Avg'] = df['Volume_Adjusted_Range'].rolling(window=28).mean()
    
    # Calculate High-Low Range Momentum
    alpha = 2 / (5 + 1)
    df['EMA_HL_Range'] = df['Daily_Price_Range'].ewm(alpha=alpha, adjust=False).mean()
    df['Daily_Change_in_EMA'] = df['EMA_HL_Range'].diff()
    
    # Calculate Volume Shock
    df['Volume_Percentage_Change'] = df['Volume'].pct_change()
    volume_threshold = 0.5
    df['Volume_Shock'] = df['Volume_Percentage_Change'].apply(lambda x: 1 if abs(x) > volume_threshold else 0)
    
    # Assess Volatility Adjustment
    df['Daily_Range'] = df['High'] - df['Low']
    df['Volatility_Score'] = df['Daily_Range'].rolling(window=30).sum() / 30
    
    # Incorporate Volume Influence
    df['EMA_Volume_5'] = df['Volume'].ewm(span=5, adjust=False).mean()
    df['EMA_Volume_10'] = df['Volume'].ewm(span=10, adjust=False).mean()
    df['Volume_Deviation_Score'] = df['Volume'] - df['Volume'].rolling(window=30).mean()
    
    # Combine Indicators
    df['Momentum_By_Volume_Shock'] = df['Daily_Change_in_EMA'] * df['Volume_Shock']
    df['Adjusted_Momentum'] = df['Momentum_By_Volume_Shock'] / df['Volatility_Score']
    df['Adjusted_Momentum'] = df['Adjusted_Momentum'] * df['Volume_Deviation_Score']
    
    # Calculate Enhanced Intraday Spread
    df['Enhanced_Intraday_Spread'] = (df['High'] - df['Low']) + (df['Close'] - df['Open'])
    
    # Adjust by Short-Term and Long-Term Momentum
    df['EMA_Close_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_Close_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Short_Term_Price_Momentum'] = df['EMA_Close_5'] - df['EMA_Close_10']
    
    df['EMA_Close_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_Close_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Long_Term_Price_Momentum'] = df['EMA_Close_20'] - df['EMA_Close_50']
    
    # Integrate Momentum and Volume Indicators
    df['Positive_Volume_Days'] = df['Volume_Percentage_Change'].apply(lambda x: 1 if x > 0 else 0)
    df['Negative_Volume_Days'] = df['Volume_Percentage_Change'].apply(lambda x: 1 if x < 0 else 0)
    df['Combined_Momentum'] = (df['Short_Term_Price_Momentum'] * df['Positive_Volume_Days']) + \
                              (df['Long_Term_Price_Momentum'] * df['Negative_Volume_Days'])
    
    # Final Alpha Factor
    df['Integrated_Factor'] = (df['Short_Term_Rolling_Avg'] - df['Long_Term_Rolling_Avg']) * df['Adjusted_Momentum']
    df['Alpha_Factor'] = df['Integrated_Factor'] * (df['Enhanced_Intraday_Spread'] * df['Volume'])
    df['Final_Alpha_Factor'] = df['Alpha_Factor'] / (df['Short_Term_Price_Momentum'] * df['Long_Term_Price_Momentum'])
    
    return df['
