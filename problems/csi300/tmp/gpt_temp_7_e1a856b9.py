import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Intraday Momentum and Volume-Weighted Price Trends
    df['High_Low_Ratio'] = df['high'] / df['low']
    df['Open_Close_Diff'] = df['close'] - df['open']
    df['Intraday_Momentum'] = (df['High_Low_Ratio'] + df['Open_Close_Diff']) * df['volume']
    
    # Calculate VWAP over varying intervals (e.g., 10 days)
    df['VWAP'] = (df['amount'].rolling(window=10).sum() / df['volume'].rolling(window=10).sum()).shift(1)
    df['VWAP_Diff'] = df['close'] - df['VWAP']
    
    # Analyze Day-to-Day and Long-Term Momentum
    df['Daily_Momentum'] = df['open'] - df['close'].shift(1)
    df['Short_Term_Momentum'] = df['close'] - df['close'].rolling(window=7).mean()
    df['Long_Term_Momentum'] = df['close'] - df['close'].rolling(window=25).mean()
    df['Momentum_Differential'] = df['Long_Term_Momentum'] - df['Short_Term_Momentum']
    
    # Evaluate Price Range and Volatility
    df['True_Range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['ATR'] = df['True_Range'].rolling(window=14).mean()
    df['ATR_Factor'] = df['ATR'] / (df['high'] - df['low'])
    
    # Examine Volume Patterns and Gaps
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Cumulative_Volume'] = df['volume'].rolling(window=10).sum()
    df['Volume_Spike'] = df['volume'] > df['volume'].rolling(window=20).mean() * 1.5
    df['Gap_Size'] = (df['open'] - df['close'].shift(1)).abs()
    
    # Analyze Open to Close Price Movement
    df['Open_Close_Percent'] = (df['close'] - df['open']) / df['open']
    df['Price_Movement_Consistency'] = df['Open_Close_Percent'].rolling(window=5).std()
    
    # Integrate Signals
    df['Volume_Adjusted_Momentum'] = df['volume'] * df['Momentum_Differential']
    df['Significant_Volume_Increase'] = df['volume'] > df['volume'].rolling(window=20).mean() * 1.5
    df.loc[df['Significant_Volume_Increase'], 'Boosted_Momentum'] = df['Volume_Adjusted_Momentum'] * 1.5
    df.loc[~df['Significant_Volume_Increase'], 'Boosted_Momentum'] = df['Volume_Adjusted_Momentum']
    df['Final_Factor'] = df['Intraday_Momentum'] + df['Boosted_Momentum']
    
    return df['Final_Factor']

# Example usage:
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(df)
