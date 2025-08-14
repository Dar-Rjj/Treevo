import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Adjusted High-Low Spread
    df['High_Low_Spread'] = df['high'] - df['low']
    df['Adjusted_High_Low_Spread'] = df['High_Low_Spread'].apply(lambda x: x if df['close'] > df['open'] else -x)
    
    # Calculate Volume-Adjusted Daily Returns
    df['Daily_Return'] = df['close'] - df['close'].shift(1)
    df['Volume_Moving_Avg'] = df['volume'].rolling(window=14).mean()
    df['Volume_Adjusted_Return'] = df.apply(lambda row: row['Daily_Return'] * 2 if row['volume'] > 1.5 * row['Volume_Moving_Avg'] else row['Daily_Return'], axis=1)
    
    # Integrate Adjusted High-Low Spread and Volume-Adjusted Returns
    df['Integrated_Factor'] = df['Adjusted_High_Low_Spread'] * df['Volume_Adjusted_Return']
    
    # Compute Volume-Adjusted Momentum
    df['Momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['Volume_Adjusted_Momentum'] = df['Momentum'] * df['volume']
    
    # Calculate Combined Intraday and Opening Gap
    df['Intraday_Volume_Weighted'] = (df['high'] - df['low']) * df['volume']
    df['Opening_Gap_Volume_Weighted'] = (df['open'] - df['close'].shift(1)) * df['volume']
    df['Combined_Value'] = df['Intraday_Volume_Weighted'] + df['Opening_Gap_Volume_Weighted']
    
    # Short-Term and Long-Term EMAs of Combined Value
    df['EMA_12'] = df['Combined_Value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Combined_Value'].ewm(span=26, adjust=False).mean()
    
    # Calculate Divergence
    df['Divergence'] = df['EMA_12'] - df['EMA_26']
    
    # Apply Sign Function to Divergence
    df['Divergence_Sign'] = df['Divergence'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # Combine Metrics
    df['Integrated_Factor_With_Momentum'] = df['Integrated_Factor'] * df['Volume_Adjusted_Momentum']
    df['Final_Alpha_Factor'] = df['Integrated_Factor_With_Momentum'] + df['Divergence_Sign']
    
    return df['Final_Alpha_Factor']
