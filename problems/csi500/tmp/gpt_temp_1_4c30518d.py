import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df, n=5, m=20, k=10, p=10):
    # Calculate the n-day moving average of closes
    df['close_ma_n'] = df['close'].rolling(window=n).mean()
    # Compute the difference between today's close and the n-day moving average of closes
    df['momentum_indicator'] = df['close'] - df['close_ma_n']
    
    # Identify days where volume exceeds the 90th percentile of the previous m days' volumes
    df['volume_90th_percentile'] = df['volume'].rolling(window=m).quantile(0.9)
    df['is_high_volume_day'] = (df['volume'] > df['volume_90th_percentile']).astype(int)
    
    # Calculate the average close on high-volume days over the last k periods
    df['high_volume_close_avg'] = (df['close'] * df['is_high_volume_day']).rolling(window=k).sum() / df['is_high_volume_day'].rolling(window=k).sum()
    df['high_volume_close_avg'].fillna(0, inplace=True)  # Handle NaNs if there are no high-volume days in the window
    
    # Calculate the daily range (High - Low)
    df['daily_range'] = df['high'] - df['low']
    
    # Compute the ratio of Close to Open, to identify bullish or bearish intraday tendencies
    df['close_to_open_ratio'] = df['close'] / df['open']
    
    # Determine if there is a gap up or down by comparing today's open with yesterday's close
    df['gap'] = df['open'] - df['close'].shift(1)
    
    # Compare the current close with the 20-day and 50-day simple moving averages
    df['close_ma_20'] = df['close'].rolling(window=20).mean()
    df['close_ma_50'] = df['close'].rolling(window=50).mean()
    df['close_vs_ma_20'] = df['close'] - df['close_ma_20']
    df['close_vs_ma_50'] = df['close'] - df['close_ma_50']
    
    # Calculate the standard deviation of returns over the past n days as a measure of recent price fluctuations
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=n).std()
    
    # Compute the correlation between daily returns and volume over the past p days
    df['return_vol_corr'] = df[['returns', 'volume']].rolling(window=p).corr().unstack().iloc[::2, :].values[:, 1]
    
    # Measure the difference between today's open and the previous day's close
    df['open_minus_prev_close'] = df['open'] - df['close'].shift(1)
    
    # Generate the final alpha factor
    df['alpha_factor'] = (
        0.2 * df['momentum_indicator'] +
        0.15 * df['high_volume_close_avg'] +
        0.1 * df['daily_range'] +
        0.1 * df['close_to_open_ratio'] +
        0.1 * df['gap'] +
        0.1 * df['close_vs_ma_20'] +
        0.1 * df['close_vs_ma_50'] +
        0.05 * df['volatility'] +
        0.05 * df['return_vol_corr'] +
        0.05 * df['open_minus_prev_close']
    )
    
    return df['alpha_factor']
