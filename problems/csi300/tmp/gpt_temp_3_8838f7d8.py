import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Determine Volatility
    df['Volatility'] = df[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)
    
    # Adjust Window Size based on Volatility
    volatility_threshold = df['Volatility'].quantile(0.5)
    df['Adaptive_Window'] = np.where(df['Volatility'] > volatility_threshold, 10, 30)
    
    # Liquidity Measures
    df['Average_Volume'] = df['volume'].rolling(window=20).mean()
    df['Dollar_Volume'] = df['close'] * df['volume']
    df['Average_Dollar_Volume'] = df['Dollar_Volume'].rolling(window=20).mean()
    
    # Adjust Volume Weighted Close-to-Open Return
    df['Adjusted_Volume_Weighted_Return'] = (df['Volume_Weighted_Return'] 
                                             / df['Average_Volume'] 
                                             / df['Average_Dollar_Volume'])
    
    # Cross-Asset Correlation
    # Assuming we have a basket of related assets
    # For simplicity, we'll use the same asset for demonstration
    close_prices = df['close']
    correlation_matrix = close_prices.rolling(window=20).corr(close_prices)
    
    # Weigh by Correlation
    df['Correlation_Adjusted_Return'] = df['Adjusted_Volume_Weighted_Return'] * correlation_matrix
    
    # Rolling Statistics
    df['Rolling_Mean'] = df['Correlation_Adjusted_Return'].rolling(window=df['Adaptive_Window']).mean()
    df['Rolling_Std'] = df['Correlation_Adjusted_Return'].rolling(window=df['Adaptive_Window']).std()
    
    # Final Factor
    df['Alpha_Factor'] = (df['Correlation_Adjusted_Return'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    return df['Alpha_Factor']
