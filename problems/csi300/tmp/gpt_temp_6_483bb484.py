import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Intraday Volatility (last 25 days' average intraday range)
    df['intraday_volatility'] = df['intraday_range'].rolling(window=25).mean()
    
    # Calculate Daily Return
    df['daily_return'] = (df['close'] / df['close'].shift(1) - 1)
    
    # Calculate Adjusted Momentum (sum of the last 22 days' daily returns)
    df['adjusted_momentum'] = df['daily_return'].rolling(window=22).sum()
    
    # Divide by Intraday Volatility to get the final adjusted momentum score
    df['adjusted_momentum'] /= df['intraday_volatility']
    
    # Incorporate Enhanced Market Sentiment
    df['market_return'] = (df['close'] / df['close'].shift(1) - 1)
    df['adjusted_momentum'] = df.apply(lambda row: row['adjusted_momentum'] * 1.10 if row['market_return'] > 0.01 else 
                                       (row['adjusted_momentum'] * 0.90 if row['market_return'] < -0.01 else row['adjusted_momentum']), axis=1)
    
    # Incorporate Volume Consideration
    df['average_volume'] = df['volume'].rolling(window=25).mean()
    df['adjusted_momentum'] = df.apply(lambda row: row['adjusted_momentum'] * 1.05 if row['volume'] > 1.5 * row['average_volume'] else 
                                       (row['adjusted_momentum'] * 0.95 if row['volume'] < 0.8 * row['average_volume'] else row['adjusted_momentum']), axis=1)
    
    # Incorporate Moving Averages
    df['50_day_ma'] = df['close'].rolling(window=50).mean()
    df['200_day_ma'] = df['close'].rolling(window=200).mean()
    df['adjusted_momentum'] = df.apply(lambda row: row['adjusted_momentum'] * 1.05 if row['close'] > row['50_day_ma'] else 
                                       (row['adjusted_momentum'] * 0.95 if row['close'] < row['200_day_ma'] else row['adjusted_momentum']), axis=1)
    
    # Incorporate Sector Performance
    # Assuming `sector_close` is a column in the DataFrame representing the sector index close price
    df['sector_return'] = (df['sector_close'] / df['sector_close'].shift(1) - 1)
    df['adjusted_momentum'] = df.apply(lambda row: row['adjusted_momentum'] * 1.05 if row['sector_return'] > 0.01 else 
                                       (row['adjusted_momentum'] * 0.95 if row['sector_return'] < -0.01 else row['adjusted_momentum']), axis=1)
    
    return df['adjusted_momentum']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
