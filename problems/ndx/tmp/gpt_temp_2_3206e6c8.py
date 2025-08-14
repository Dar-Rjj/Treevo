import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate daily, weekly, and monthly percentage change in closing prices
    df['daily_return'] = df['close'].pct_change()
    df['weekly_return'] = df['close'].pct_change(5)
    df['monthly_return'] = df['close'].pct_change(20)
    
    # Calculate 10-day and 20-day moving averages of daily returns
    df['10_day_ma'] = df['daily_return'].rolling(window=10).mean()
    df['20_day_ma'] = df['daily_return'].rolling(window=20).mean()
    
    # Determine current price relative to moving averages
    df['price_above_10_20_MA'] = (df['close'] > df['10_day_ma']) & (df['close'] > df['20_day_ma'])
    df['price_below_10_20_MA'] = (df['close'] < df['10_day_ma']) & (df['close'] < df['20_day_ma'])
    
    # Calculate volume ratios
    df['volume_ratio_5_day'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['volume_ratio_20_day'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Calculate correlation between daily volume and price changes
    df['volume_price_corr'] = df[['volume', 'daily_return']].rolling(window=20).corr().unstack().iloc[::2, :].iloc[:, 1]
    
    # Sum of volume for positive and negative return days
    df['positive_volume_sum'] = df[df['daily_return'] > 0]['volume'].rolling(window=20).sum()
    df['negative_volume_sum'] = df[df['daily_return'] < 0]['volume'].rolling(window=20).sum()
    
    # Intraday range as a percentage of previous close
    df['intraday_range_pct'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Ratio of open-close difference to intraday range
    df['open_close_diff_intraday_range'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Intraday movement direction
    df['intraday_movement'] = (df['close'] > df['open']).astype(int)
    
    # Momentum-based signal
    df['momentum_signal'] = ((df['daily_return'] > 0) & (df['10_day_ma'] > df['20_day_ma'])).astype(int)
    
    # Volume-based signal
    df['volume_signal'] = ((df['volume_ratio_5_day'] > 1.5) & (df['daily_return'] > 0)).astype(int) - \
                          ((df['volume_ratio_5_day'] < 0.5) & (df['daily_return'] < 0)).astype(int)
    
    # Intraday price movement signal
    df['intraday_signal'] = ((df['intraday_range_pct'] > 0.03) & (df['intraday_movement'] == 1)).astype(int) - \
                            ((df['intraday_range_pct'] > 0.03) & (df['intraday_movement'] == 0)).astype(int)
    
    # Composite score
    df['composite_score'] = df['momentum_signal'] + df['volume_signal'] + df['intraday_signal']
    
    return df['composite_score']
