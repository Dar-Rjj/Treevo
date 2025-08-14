import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20, m=30, p=10, q=5, r=14):
    # Calculate the difference between close and open prices (price action)
    df['price_action'] = df['close'] - df['open']
    
    # Rolling window of price action to find trends in market sentiment
    df['price_action_trend'] = df['price_action'].rolling(window=n).mean()
    
    # Daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Average trading volume over the last n days
    df['avg_volume'] = df['volume'].rolling(window=n).mean()
    
    # Factor by dividing daily return by average trading volume
    df['return_by_volume'] = df['daily_return'] / df['avg_volume']
    
    # 50-day simple moving average (SMA) and 200-day SMA
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Difference between 50-day SMA and 200-day SMA
    df['sma_diff'] = df['sma_50'] - df['sma_200']
    
    # Volatility metric from the standard deviation of daily returns
    df['volatility'] = df['daily_return'].rolling(window=m).std()
    
    # Maximum high-low range in the previous p days
    df['high_low_range'] = df['high'] - df['low']
    df['max_high_low_range'] = df['high_low_range'].rolling(window=p).max()
    
    # Cumulative sum of positive and negative price changes over q days
    df['pos_changes'] = df['close'].diff().apply(lambda x: max(x, 0))
    df['neg_changes'] = df['close'].diff().apply(lambda x: min(x, 0))
    df['cum_pos_changes'] = df['pos_changes'].rolling(window=q).sum()
    df['cum_neg_changes'] = df['neg_changes'].rolling(window=q).sum()
    
    # Measure the distance between the closing price and the next day's opening price
    df['gap'] = df['open'].shift(-1) - df['close']
    
    # Average true range (ATR) calculated over r days
    df['tr'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['atr'] = df['tr'].rolling(window=r).mean()
    
    # Gap size against the ATR
    df['scaled_gap'] = df['gap'] / df['atr']
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['price_action_trend'] + df['return_by_volume'] + 
                          df['sma_diff'] - df['volatility'] + df['max_high_low_range'] + 
                          df['cum_pos_changes'] - df['cum_neg_changes'] + df['scaled_gap']).fillna(0)
    
    return df['alpha_factor']
