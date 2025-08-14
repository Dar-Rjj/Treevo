import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate the 5-day percentage change in close price
    df['pct_change_5d'] = df['close'].pct_change(periods=5)
    
    # Calculate the 14-day Average True Range (ATR)
    df['h_l'] = df['high'] - df['low']
    df['h_pc'] = np.abs(df['high'] - df['close'].shift(1))
    df['l_pc'] = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    df['atr_14d'] = df['tr'].rolling(window=14).mean()
    
    # Calculate the 20-day standard deviation of daily returns
    df['daily_returns'] = df['close'].pct_change()
    df['std_20d'] = df['daily_returns'].rolling(window=20).std()
    
    # Calculate the ratio of current day's volume to the 30-day moving average volume
    df['vol_ma_30d'] = df['volume'].rolling(window=30).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma_30d']
    
    # Calculate the 10-day change in On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_10d'] = df['obv'].diff(periods=10)
    
    # Calculate the 20-day Donchian Channel width
    df['donchian_high_20d'] = df['high'].rolling(window=20).max()
    df['donchian_low_20d'] = df['low'].rolling(window=20).min()
    df['donchian_width_20d'] = df['donchian_high_20d'] - df['donchian_low_20d']
    
    # Calculate the difference between today’s high and yesterday’s high
    df['high_diff_yesterday'] = df['high'] - df['high'].shift(1)
    
    # Calculate the difference between today’s low and yesterday’s low
    df['low_diff_yesterday'] = df['low'] - df['low'].shift(1)
    
    # Calculate the gap between the opening and closing prices
    df['open_close_gap'] = df['open'] - df['close']
    
    # Calculate the intraday price range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate the 10-day change in Accumulation/Distribution Line (A/D Line)
    df['ad_line'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']
    df['ad_line_10d'] = df['ad_line'].diff(periods=10)
    
    # Identify days with unusually high volume
    df['volume_spike'] = (df['volume'] > df['volume'].rolling(window=30).mean() + 2 * df['volume'].rolling(window=30).std()).astype(int)
    
    # Use the correlation between volume and price change over a 10-day period as an alpha factor
    df['price_vol_corr_10d'] = df['close'].pct_change().rolling(window=10).corr(df['volume'])
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df[['pct_change_5d', 'atr_14d', 'std_20d', 'vol_ratio', 'obv_10d', 'donchian_width_20d', 
               'high_diff_yesterday', 'low_diff_yesterday', 'open_close_gap', 'intraday_range', 
               'ad_line_10d', 'volume_spike', 'price_vol_corr_10d']]
