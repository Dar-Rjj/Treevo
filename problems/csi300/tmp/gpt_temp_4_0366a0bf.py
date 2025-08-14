import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Volatility
    df['Intraday_Volatility'] = df['high'] - df['low']
    
    # Weight by Volume
    df['Weighted_Volatility'] = df['Intraday_Volatility'] * df['volume']
    
    # Calculate Intraday Momentum
    df['Close_Open_Difference'] = df['close'] - df['open']
    
    # Define rolling windows
    short_term_window = 5
    long_term_window = 20
    
    # Calculate Short-Term and Long-Term Rolling Sums
    df['Intraday_Momentum_Short_Term'] = df['Close_Open_Difference'].rolling(window=short_term_window).sum()
    df['Intraday_Momentum_Long_Term'] = df['Close_Open_Difference'].rolling(window=long_term_window).sum()
    
    # Integrate Momentum and Volatility
    df['Integrated_Value_Short_Term'] = df['Weighted_Volatility'] + df['Intraday_Momentum_Short_Term']
    df['Integrated_Value_Long_Term'] = df['Weighted_Volatility'] + df['Intraday_Momentum_Long_Term']
    
    # Apply Dynamic Exponential Smoothing
    short_term_smoothing_factor = 0.8
    long_term_smoothing_factor = 0.95
    
    df['Smoothed_Value_Short_Term'] = df['Integrated_Value_Short_Term'].ewm(alpha=short_term_smoothing_factor).mean()
    df['Smoothed_Value_Long_Term'] = df['Integrated_Value_Long_Term'].ewm(alpha=long_term_smoothing_factor).mean()
    
    # Ensure Values are Positive
    small_constant = 1e-6
    df['Positive_Smoothed_Value_Short_Term'] = df['Smoothed_Value_Short_Term'] + small_constant
    df['Positive_Smoothed_Value_Long_Term'] = df['Smoothed_Value_Long_Term'] + small_constant
    
    # Apply Logarithmic Transformation
    df['Log_Transformed_Value_Short_Term'] = np.log(df['Positive_Smoothed_Value_Short_Term'])
    df['Log_Transformed_Value_Long_Term'] = np.log(df['Positive_Smoothed_Value_Long_Term'])
    
    # Factor Output
    factor_series_short_term = df['Log_Transformed_Value_Short_Term']
    factor_series_long_term = df['Log_Transformed_Value_Long_Term']
    
    # Return the integrated and transformed intraday momentum and volatility
    return pd.Series(factor_series_short_term, name='Short_Term_Factor'), pd.Series(factor_series_long_term, name='Long_Term_Factor')
