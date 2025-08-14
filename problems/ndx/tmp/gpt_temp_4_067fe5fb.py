import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate percentage change in closing prices over daily, weekly, and monthly periods
    df['daily_return'] = df['close'].pct_change()
    df['weekly_return'] = df['close'].pct_change(periods=5)
    df['monthly_return'] = df['close'].pct_change(periods=20)
    
    # Calculate 10-day and 20-day moving averages of daily returns
    df['10_day_ma'] = df['daily_return'].rolling(window=10).mean()
    df['20_day_ma'] = df['daily_return'].rolling(window=20).mean()
    
    # Determine if the current price is above or below these moving averages
    df['price_above_10_ma'] = (df['close'] > df['close'].shift(10).rolling(window=10).mean()).astype(int)
    df['price_above_20_ma'] = (df['close'] > df['close'].shift(20).rolling(window=20).mean()).astype(int)
    
    # Analyze the ratio of today's volume to 5-day and 20-day average volumes
    df['volume_ratio_5d'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['volume_ratio_20d'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Calculate the correlation between daily volume changes and daily price changes
    df['vol_chg'] = df['volume'].pct_change()
    df['price_vol_corr'] = df[['daily_return', 'vol_chg']].rolling(window=20).corr().iloc[::2, -1]
    
    # Sum of volume for days with positive and negative returns
    df['pos_vol_sum'] = df[df['daily_return'] > 0]['volume']
    df['neg_vol_sum'] = df[df['daily_return'] < 0]['volume']
    df['pos_vol_sum'] = df['pos_vol_sum'].fillna(0).rolling(window=20).sum()
    df['neg_vol_sum'] = df['neg_vol_sum'].fillna(0).rolling(window=20).sum()
    
    # Calculate the range (high - low) as a percentage of the previous day's close
    df['intraday_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Calculate the ratio of the difference between open and close to the range (high - low)
    df['open_close_range_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Mark "up" day if (close > open), "down" day if (close < open)
    df['up_day'] = (df['close'] > df['open']).astype(int)
    
    # Create a momentum-based signal
    df['momentum_signal'] = 0
    df.loc[(df['daily_return'] > 0) & (df['10_day_ma'] > df['20_day_ma']), 'momentum_signal'] = 1
    df.loc[(df['daily_return'] < 0) & (df['10_day_ma'] < df['20_day_ma']), 'momentum_signal'] = -1
    
    # Create a volume-based signal
    df['volume_signal'] = 0
    df.loc[(df['volume_ratio_5d'] > 1.5) & (df['daily_return'] > 0), 'volume_signal'] = 1
    df.loc[(df['volume_ratio_5d'] < 0.5) & (df['daily_return'] < 0), 'volume_signal'] = -1
    
    # Create an intraday price movement signal
    df['intraday_signal'] = 0
    df.loc[(df['intraday_range'] > 0.03) & (df['up_day'] == 1), 'intraday_signal'] = 1
    df.loc[(df['intraday_range'] > 0.03) & (df['up_day'] == 0), 'intraday_signal'] = -1
    
    # Combine all signals into a composite alpha factor
    df['composite_alpha'] = df['momentum_signal'] + df['volume_signal'] + df['intraday_signal']
    
    return df['composite_alpha']
