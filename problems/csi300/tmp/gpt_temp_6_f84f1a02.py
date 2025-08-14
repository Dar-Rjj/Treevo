import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Return
    df['Daily_Return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Volume-Weighted Momentum
    df['Volume_Weighted_Momentum'] = (df['Daily_Return'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Calculate Volume Shock Indicator
    df['Volume_Shock'] = df['volume'] > 2 * df['volume'].rolling(window=20).mean()
    
    # Adjust Volume-Weighted Momentum
    fixed_factor = 1.1  # Example fixed factor
    df['Adjusted_Volume_Weighted_Momentum'] = df['Volume_Weighted_Momentum'] * (fixed_factor if df['Volume_Shock'] else 1)
    
    # Calculate Daily Price Movement
    df['Price_Movement'] = df['close'] - df['close'].shift(1)
    
    # Calculate Intraday Volatility
    df['Intraday_Volatility'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Price Change Over Time Window
    df['Price_Change_20'] = df['close'] - df['close'].shift(20)
    
    # Calculate Historical Price Volatility
    df['Historical_Volatility'] = df['close'].rolling(window=20).std()
    
    # Calculate Volume Direction
    df['Volume_Direction'] = (df['volume'] > df['volume'].shift(1)).astype(int) * 2 - 1
    
    # Combine Price Movement and Volume Direction
    df['Combined_Price_Volume'] = df['Price_Movement'] * df['Volume_Direction']
    
    # Weight by Volume and Inverse Historical Volatility
    df['Inverse_Historical_Volatility'] = 1 / df['Historical_Volatility']
    df['Combined_Weights'] = df['volume'] * df['Inverse_Historical_Volatility']
    
    # Calculate Volume-to-Price Ratio
    df['Volume_to_Price_Ratio'] = df['volume'] / df['close']
    
    # Final Factor
    df['Final_Factor'] = (
        df['Adjusted_Volume_Weighted_Momentum'] * 
        df['Combined_Weights'] * 
        df['Price_Change_20'] + 
        df['Combined_Price_Volume'] + 
        df['Intraday_Volatility'] * df['Volume_to_Price_Ratio'] - 
        (df['close'] - df['open']).pow(2)
    ).rolling(window=20).mean()
    
    return df['Final_Factor']
