import pandas as pd
import pandas as pd

def heuristics_v2(df, N=14):
    # Calculate True Range
    df['TrueRange'] = df[['High', 'Low']].apply(lambda x: max(x['High'] - x['Low'], 
                                                               abs(x['High'] - df['Close'].shift(1)), 
                                                               abs(x['Low'] - df['Close'].shift(1))), axis=1)
    
    # Compute Average True Range (ATR)
    df['ATR'] = df['TrueRange'].rolling(window=N).mean()
    
    # Calculate Price Momentum
    df['PriceMomentum'] = df['Close'] - df['Close'].shift(N)
    
    # Adjust Momentum by ATR
    df['AdjustedMomentum'] = df['PriceMomentum'] / df['ATR']
    
    # Introduce Volume Adjustment
    df['VolumeMA'] = df['Volume'].rolling(window=N).mean()
    df['Factor'] = df['AdjustedMomentum'] * df['VolumeMA']
    
    return df['Factor']

# Example usage:
# df = pd.DataFrame({
#     'Open': [100, 101, 102, ...],
#     'High': [105, 106, 107, ...],
#     'Low': [95, 96, 97, ...],
#     'Close': [103, 104, 105, ...],
#     'Amount': [1000, 1000, 1000, ...],
#     'Volume': [10000, 10000, 10000, ...]
# }, index=pd.date_range(start='2023-01-01', periods=len(data)))
# factor = heuristics_v2(df)
