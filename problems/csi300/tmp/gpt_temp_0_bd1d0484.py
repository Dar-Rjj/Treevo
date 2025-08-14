import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 14-day and 50-day simple moving averages (SMA)
    df['SMA_14'] = df['close'].rolling(window=14).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Momentum crossover signal
    df['momentum_crossover'] = df['SMA_50'] - df['SMA_14']
    df['momentum_indicator'] = df['momentum_crossover'].apply(lambda x: 1 if x > 0 else -1)
    
    # Calculate 21-day standard deviation of closing prices
    df['std_21'] = df['close'].rolling(window=21).std()
    
    # Calculate 21-day average of daily returns
    df['daily_return'] = df['close'].pct_change()
    df['avg_return_21'] = df['daily_return'].rolling(window=21).mean()
    
    # Volatility measure
    df['volatility_ratio'] = df['std_21'] / df['avg_return_21']
    df['inverse_volatility_ratio'] = 1 / df['volatility_ratio']
    
    # 7-day moving average of volume
    df['volume_MA_7'] = df['volume'].rolling(window=7).mean()
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume_MA_7']
    
    # Identify engulfing patterns
    def engulfing_pattern(row, prev_row):
        if (row['open'] < row['close']) and (prev_row['open'] > prev_row['close']):
            if (row['close'] > prev_row['open']) and (row['open'] < prev_row['close']):
                return 1  # Bullish engulfing
        elif (row['open'] > row['close']) and (prev_row['open'] < prev_row['close']):
            if (row['close'] < prev_row['open']) and (row['open'] > prev_row['close']):
                return -1  # Bearish engulfing
        return 0
    
    df['engulfing_score'] = df.apply(lambda row: engulfing_pattern(row, df.shift(1).iloc[row.name]) if row.name > 0 else 0, axis=1)
    df['engulfing_trend'] = df['engulfing_score'].rolling(window=10).sum()
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['momentum_indicator'] + df['inverse_volatility_ratio'] + df['volume_ratio'] + df['engulfing_trend'])
    
    return df['alpha_factor']
