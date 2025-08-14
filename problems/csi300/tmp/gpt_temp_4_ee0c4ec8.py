import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate short-term EMA (5 days) and long-term EMA (20 days)
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Apply EMA to sentiment (assuming 'sentiment' is a column in the DataFrame)
    df['Sentiment_EMA_5'] = df['sentiment'].ewm(span=5, adjust=False).mean()
    df['Sentiment_EMA_20'] = df['sentiment'].ewm(span=20, adjust=False).mean()
    
    # Volume-Weighted Close Price
    total_volume = df['volume'].rolling(window=5).sum()
    df['VW_Close_5'] = (df['close'] * df['volume']).rolling(window=5).sum() / total_volume
    df['VW_Close_20'] = (df['close'] * df['volume']).rolling(window=20).sum() / total_volume
    
    # Volume-Weighted High-Low Difference
    df['VW_HL_5'] = ((df['high'] - df['low']) * df['volume']).rolling(window=5).sum()
    df['VW_HL_20'] = ((df['high'] - df['low']) * df['volume']).rolling(window=20).sum()
    
    # Time-Series Momentum
    df['Momentum'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['Weight_Short'] = np.where(df['Momentum'] > 0, 0.7, 0.3)
    
    # Sector Rotation (assuming 'sector' and 'sector_performance' are columns in the DataFrame)
    leading_sector = df.groupby('date')['sector_performance'].transform('max')
    df['Weight_Leading'] = np.where(df['sector'] == leading_sector, 0.8, 0.2)
    
    # Exponential Smoothing
    df['EMA_5_Smooth'] = df['EMA_5'] * 0.8 + df['EMA_5'].shift(1) * 0.2
    df['EMA_20_Smooth'] = df['EMA_20'] * 0.8 + df['EMA_20'].shift(1) * 0.2
    
    # Combine factors with dynamic weighting
    df['Factor'] = (
        df['Weight_Short'] * (df['EMA_5_Smooth'] + df['Sentiment_EMA_5'] + df['VW_Close_5'] - df['VW_HL_5']) +
        (1 - df['Weight_Short']) * (df['EMA_20_Smooth'] + df['Sentiment_EMA_20'] + df['VW_Close_20'] - df['VW_HL_20'])
    ) * df['Weight_Leading']
    
    return df['Factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor = heuristics_v2(df)
