import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the 5-day and 20-day moving averages of closing prices
    df['5_day_ma'] = df['close'].rolling(window=5).mean()
    df['20_day_ma'] = df['close'].rolling(window=20).mean()
    
    # Calculate the difference between short-term and long-term moving averages
    df['ma_diff'] = df['5_day_ma'] - df['20_day_ma']
    
    # Compute the 14-day exponential moving average (EMA) of the closing price
    df['14_day_ema'] = df['close'].ewm(span=14, adjust=False).mean()
    
    # Find the difference between the current close and the 14-day EMA
    df['close_ema_diff'] = df['close'] - df['14_day_ema']
    
    # Develop an on-balance volume (OBV) indicator adjusted by the closing price
    df['obv'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['obv'] = df['obv'].cumsum()
    df['obv_adj_close'] = df['obv'] / df['close']
    
    # Measure the ratio of the volume of days with positive return over the total volume within the past 30 trading days
    df['positive_return'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['positive_vol_ratio'] = df['positive_return'] * df['volume'] / df['volume'].rolling(window=30).sum()
    df['positive_vol_ratio'] = df['positive_vol_ratio'].rolling(window=30).sum()
    
    # Construct a true range (TR) by taking the maximum of [high - low, abs(high - previous close), abs(low - previous close)]
    df['tr'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    
    # Compute the average true range (ATR) over the last 14 days
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Estimate the volatility of daily returns by calculating the standard deviation of close-to-close returns over the most recent 30 days
    df['daily_return'] = df['close'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=30).std()
    
    # Identify up gaps (open > previous close) and down gaps (open < previous close)
    df['up_gap'] = (df['open'] > df['close'].shift(1)).astype(int)
    df['down_gap'] = (df['open'] < df['close'].shift(1)).astype(int)
    
    # Count their occurrences in the past 30 days
    df['up_gap_count'] = df['up_gap'].rolling(window=30).sum()
    df['down_gap_count'] = df['down_gap'].rolling(window=30).sum()
    
    # Calculate the sum of the absolute values of up gaps and down gaps separately for the last 30 days
    df['up_gap_sum'] = (df['open'] - df['close'].shift(1)) * df['up_gap']
    df['down_gap_sum'] = (df['close'].shift(1) - df['open']) * df['down_gap']
    df['up_gap_sum'] = df['up_gap_sum'].rolling(window=30).sum()
    df['down_gap_sum'] = df['down_gap_sum'].rolling(window=30).sum()
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['ma_diff'] + df['close_ema_diff'] + df['obv_adj_close'] + df['positive_vol_ratio'] - df['atr'] - df['volatility'] + df['up_gap_count'] - df['down_gap_count'] + df['up_gap_sum'] - df['down_gap_sum'])
    
    return df['alpha_factor']
