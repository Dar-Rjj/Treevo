import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate simple moving averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Calculate exponential moving averages (EMA)
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Compute the difference between 20-day EMA and 50-day EMA as a trend strength indicator
    df['Trend_Strength'] = df['EMA_20'] - df['EMA_50']
    
    # Flag days with trading volume significantly higher than the 20-day moving average
    df['Volume_Spike'] = df['volume'] > df['volume'].rolling(window=20).mean() * 1.5
    
    # Calculate the correlation between daily volume and price changes over the past 30 days
    df['Price_Change'] = df['close'].pct_change()
    df['Volume_Price_Corr'] = df['Price_Change'].rolling(window=30).corr(df['volume'])
    
    # Calculate the daily range (high - low) and its ratio to the closing price
    df['Daily_Range'] = df['high'] - df['low']
    df['Range_Ratio'] = df['Daily_Range'] / df['close']
    
    # Measure the difference between the opening price of the current day and the closing price of the previous day
    df['Open_Gap'] = df['open'] - df['close'].shift(1)
    
    # Calculate the true range
    df['True_Range'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0], x[-1]) - min(x[0], x[-1]), axis=1)
    
    # Calculate the 14-period ATR based on the true range
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    
    # Summarize multiple indicators into a single score
    df['Composite_Factor'] = (
        df['SMA_5'] + 
        df['EMA_5'] + 
        df['Trend_Strength'] + 
        df['Volume_Spike'].astype(int) + 
        df['Volume_Price_Corr'] + 
        df['Range_Ratio'] + 
        df['Open_Gap'] + 
        df['ATR_14']
    )
    
    return df['Composite_Factor']

# Example usage:
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# composite_factor = heuristics_v2(df)
# print(composite_factor)
