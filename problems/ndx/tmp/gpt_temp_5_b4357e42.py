import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=10, percentile_threshold=0.75):
    # Calculate Intraday Range
    df['Intraday_Range'] = df['high'] - df['low']
    
    # Compute Price Change
    df['Price_Change'] = df['close'] - df['close'].shift(1)
    
    # Detect Significant Volume Increase
    df['Avg_Volume'] = df['volume'].rolling(window=N).mean()
    df['Volume_Spike'] = (df['volume'] > 2 * df['Avg_Volume']).astype(int)
    
    # Normalize Price Change by Intraday Volatility
    df['Adjusted_Price_Change'] = df['Price_Change'] / df['Intraday_Range']
    
    # Apply Volume-Weighted Adjustment
    df['Weighted_Adjusted_Price_Change'] = np.where(
        df['Volume_Spike'] == 1,
        df['volume'] * (df['Adjusted_Price_Change'] * 2),
        df['volume'] * df['Adjusted_Price_Change']
    )
    
    # Accumulate Momentum Score
    df['Momentum_Score'] = df['Weighted_Adjusted_Price_Change'].rolling(window=N).sum()
    
    # Calculate Rate of Change (ROC)
    df['ROC'] = (df['close'] - df['close'].shift(14)) / df['close'].shift(14)
    
    # Calculate Average True Range (ATR)
    df['True_Range'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['ATR'] = df['True_Range'].rolling(window=10).mean()
    
    # Combine Momentum, Volatility, and Volume
    df['Composite_Alpha_Factor'] = (df['Momentum_Score'] + df['ROC'] + df['ATR']) / 3
    
    # Apply Threshold
    threshold = df['Composite_Alpha_Factor'].quantile(percentile_threshold)
    df['Composite_Alpha_Factor'] = df['Composite_Alpha_Factor'].where(df['Composite_Alpha_Factor'] >= threshold, 0)
    
    return df['Composite_Alpha_Factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
