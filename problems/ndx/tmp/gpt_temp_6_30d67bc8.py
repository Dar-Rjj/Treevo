import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Price Difference
    df['Price_Diff'] = df['high'] - df['low']
    
    # Determine Gain and Loss
    df['Gain'] = (df['close'] > df['close'].shift(1)) * (df['close'] - df['close'].shift(1))
    df['Loss'] = (df['close'] <= df['close'].shift(1)) * (df['close'].shift(1) - df['close'])
    
    # Aggregate Gains and Losses
    df['Sum_Gain'] = df['Gain'].rolling(window=14).sum()
    df['Sum_Loss'] = df['Loss'].rolling(window=14).sum()
    
    # Calculate Relative Strength
    df['RS'] = df['Sum_Gain'] / df['Sum_Loss']
    
    # Convert to ARSI
    df['ARSI'] = 100 - (100 / (1 + df['RS']))
    df['ARSI_Adjusted'] = df['ARSI'] * df['volume'].rolling(window=14).mean() * (df['close'] / df['open'] - 1)
    
    # Calculate High-Low Spread
    df['High_Low_Spread'] = df['high'] - df['low']
    
    # Compute VWMA of High-Low Spread
    df['HL_VWMA'] = (df['High_Low_Spread'] * df['volume']).rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    
    # Incorporate Close-to-Close Return
    df['Close_to_Close_Return'] = df['close'] - df['close'].shift(1)
    df['HL_VWMA_Adjusted'] = df['HL_VWMA'] * df['Close_to_Close_Return']
    
    # Calculate Short-Term Return
    df['Short_Term_Return'] = df['close'].pct_change(periods=5)
    
    # Calculate Long-Term Return
    df['Long_Term_Return'] = df['close'].pct_change(periods=20)
    
    # Calculate Volume-Weighted Short-Term Return
    df['VW_Short_Term_Return'] = df['volume'] * df['Short_Term_Return']
    
    # Calculate Volume-Weighted Long-Term Return
    df['VW_Long_Term_Return'] = df['volume'] * df['Long_Term_Return']
    
    # Calculate Short-Term Volatility
    df['ATR'] = df[['high', 'low']].rolling(window=5).max() - df[['high', 'low']].rolling(window=5).min()
    df['Short_Term_Volatility'] = df['ATR'].mean(axis=1)
    
    # Adjust for Volatility
    df['Adjusted_VW_Short_Term_Return'] = df['VW_Short_Term_Return'] / df['Short_Term_Volatility']
    
    # Calculate Price Oscillator
    df['Price_Oscillator'] = df['close'].shift(5) - df['close'].shift(20)
    
    # Combine ARSI, Adjusted Returns, Price Oscillator, and VWMA
    df['Factor'] = (df['ARSI_Adjusted'] * df['Adjusted_VW_Short_Term_Return'] 
                    + df['Price_Oscillator'] 
                    + df['HL_VWMA_Adjusted'] 
                    - df['VW_Long_Term_Return'])
    
    return df['Factor']
