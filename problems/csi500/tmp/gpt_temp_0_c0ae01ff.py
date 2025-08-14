import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the difference between today's close and yesterday's open
    df['Close_Yesterday_Open'] = df['close'] - df['open'].shift(1)
    
    # Calculate the high-low range for volatility
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Calculate the percent change in volume from the previous day
    df['Volume_Change'] = df['volume'] / df['volume'].shift(1) - 1
    
    # Calculate the ratio of today's amount to the moving average of the past 5 days' amount
    df['Amount_Ratio'] = df['amount'] / df['amount'].rolling(window=5).mean()
    
    # Calculate the percentage change in close over the past 5 days
    df['Close_5D_Pct_Change'] = df['close'].pct_change(periods=5)
    
    # Calculate the percentage change in volume over the past 5 days
    df['Volume_5D_Pct_Change'] = df['volume'].pct_change(periods=5)
    
    # Create a momentum-based signal
    df['Momentum_Signal'] = df['Close_5D_Pct_Change'] * df['Volume_5D_Pct_Change']
    
    # Calculate the 5-day moving sum of the high-low range
    df['High_Low_5D_Sum'] = df['High_Low_Range'].rolling(window=5).sum()
    
    # Incorporate volatility into the momentum
    df['Volatility_Adjusted_Momentum'] = df['High_Low_5D_Sum'] + df['Momentum_Signal']
    
    # Calculate the slope of a linear regression line fitted to the closing prices over the past 20 days
    def calculate_slope(series):
        return pd.Series(range(1, len(series) + 1)).cov(series) / pd.Series(range(1, len(series) + 1)).var()
    
    df['Slope_20D'] = df['close'].rolling(window=20).apply(calculate_slope, raw=False)
    
    # Calculate the average volume over the same period
    df['Volume_20D_Avg'] = df['volume'].rolling(window=20).mean()
    
    # Enhance the trend signal with volume
    df['Trend_Signal'] = df['Slope_20D'] * df['Volume_20D_Avg']
    
    # Combine different indicators for a composite factor
    df['Composite_Factor'] = (df['Close_Yesterday_Open'] + 
                              df['High_Low_Range'] + 
                              df['Volume_Change'] + 
                              df['Amount_Ratio'] + 
                              df['Momentum_Signal'] + 
                              df['Volatility_Adjusted_Momentum'] + 
                              df['Trend_Signal'])
    
    return df['Composite_Factor']
