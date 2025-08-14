import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily and weekly returns
    df['daily_return'] = df['close'].pct_change()
    df['weekly_return'] = df['close'].pct_change(periods=5)
    
    # Calculate true range
    df['true_range'] = df.apply(lambda x: max(x['high'] - x['low'], 
                                              abs(x['high'] - df.loc[x.name - pd.Timedelta(days=1), 'close']), 
                                              abs(x['low'] - df.loc[x.name - pd.Timedelta(days=1), 'close'])), axis=1)
    
    # Calculate Average True Range (ATR) over the last 14 days
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    
    # Calculate volume-weighted price
    df['volume_weighted_price'] = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    
    # Measure the correlation between daily log returns and the change in volume over the last 20 trading days
    df['log_return'] = np.log(df['close']).diff()
    df['volume_change'] = df['volume'].pct_change()
    df['correlation_log_vol'] = df['log_return'].rolling(window=20).corr(df['volume_change'])
    
    # Short-term (9-day) and long-term (26-day) simple moving averages
    df['sma_9'] = df['close'].rolling(window=9).mean()
    df['sma_26'] = df['close'].rolling(window=26).mean()
    
    # Bullish and bearish crossovers
    df['crossover_signal'] = np.where(df['sma_9'] > df['sma_26'], 1, 0)
    
    # Relative Strength Index (RSI) over the last 14 days
    df['gain'] = df['close'].diff().clip(lower=0)
    df['loss'] = df['close'].diff().clip(upper=0).abs()
    df['avg_gain'] = df['gain'].rolling(window=14).mean()
    df['avg_loss'] = df['loss'].rolling(window=14).mean()
    df['rsi_14'] = 100 - (100 / (1 + (df['avg_gain'] / df['avg_loss'])))
    
    # Historical volatility using the standard deviation of daily returns over the past 20 days
    df['volatility_20'] = df['daily_return'].rolling(window=20).std()
    
    # Mean return following periods identified as having above-average volatility
    df['above_avg_vol'] = df['volatility_20'] > df['volatility_20'].mean()
    df['mean_return_high_vol'] = df[df['above_avg_vol']]['daily_return'].shift(-1).rolling(window=20).mean()
    
    # Opening gaps
    df['opening_gap'] = df['open'] - df['close'].shift(1)
    
    # Average return over the next 3, 5, and 10 trading days for stocks experiencing large gaps
    df['next_3_day_return'] = df['close'].shift(-3) / df['close'] - 1
    df['next_5_day_return'] = df['close'].shift(-5) / df['close'] - 1
    df['next_10_day_return'] = df['close'].shift(-10) / df['close'] - 1
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['weekly_return'] 
                          + df['atr_14'] 
                          + df['correlation_log_vol'] 
                          + df['crossover_signal'] 
                          + df['rsi_14'] 
                          + df['mean_return_high_vol'] 
                          + df['opening_gap'] 
                          + df['next_3_day_return'] 
                          + df['next_5_day_return'] 
                          + df['next_10_day_return'])
    
    return df['alpha_factor']
