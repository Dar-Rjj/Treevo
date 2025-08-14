import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, M=5, K=10):
    # Calculate Intraday High-Low Spread
    df['Intraday_Spread'] = df['high'] - df['low']
    
    # Compute Intraday Range Weighted Average Price (IRWAP)
    df['TotalValue'] = df['close'] * df['volume']
    df['TotalVolume'] = df['volume']
    IRWAP = df.groupby(df.index.date)['TotalValue'].sum() / df.groupby(df.index.date)['TotalVolume'].sum()
    IRWAP = IRWAP.reindex(df.index.normalize(), method='ffill')
    
    # Evaluate IRWAP Difference
    df['IRWAP_Diff'] = df['close'] - IRWAP
    
    # Calculate Adjusted Volume
    df['Adjusted_Volume'] = df['volume'] / df['Intraday_Spread']
    
    # Integrate Volume-Adjusted Log Return
    df['Log_Return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    df['Volume_Adjusted_Log_Return'] = df['Log_Return'] * df['Adjusted_Volume']
    
    # Compare Latest Volume to M-day Average
    df['M_day_Avg_Volume'] = df['volume'].rolling(window=M).mean()
    df['Volume_Score'] = np.where(df['volume'] > df['M_day_Avg_Volume'], 1, -1)
    
    # Compare Latest Amount to K-day Average
    df['K_day_Avg_Amount'] = df['amount'].rolling(window=K).mean()
    df['Amount_Score'] = np.where(df['amount'] > df['K_day_Avg_Amount'], 1, -1)
    
    # Combine all scores and adjusted returns
    df['Combined_Score'] = (df['Volume_Score'] + df['Amount_Score'] + df['Volume_Adjusted_Log_Return']) / 3

    return df['Combined_Score']

# Example usage
# df = pd.read_csv('your_data.csv', index_col=['date'], parse_dates=['date'])
# factor_values = heuristics_v2(df)
