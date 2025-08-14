import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate simple moving averages (SMA) for different periods
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # Calculate exponential moving averages (EMA) for different periods
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Compute the difference between 20-day EMA and 50-day EMA as a trend strength indicator
    df['EMA_diff'] = df['EMA_20'] - df['EMA_50']

    # Calculate the rate of change (ROC) of the 50-day EMA to identify accelerating or decelerating trends
    df['EMA_50_ROC'] = df['EMA_50'].pct_change()

    # Measure volume spikes
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > 1.5 * df['vol_ma_20']

    # Identify days with volume spikes that coincide with large price movements
    df['price_change'] = df['close'].pct_change()
    df['volume_price_spike'] = (df['volume_spike']) & (df['price_change'].abs() > 0.05)

    # Calculate the correlation between daily volume and price changes over the past 30 days
    df['vol_price_corr_30'] = df[['volume', 'price_change']].rolling(window=30).corr().iloc[::2, 1]

    # Analyze the ratio of up-volume to down-volume over the past 30 days
    df['up_volume'] = df['volume'] * (df['close'] > df['open'])
    df['down_volume'] = df['volume'] * (df['close'] < df['open'])
    df['up_down_ratio'] = df['up_volume'].rolling(window=30).sum() / df['down_volume'].rolling(window=30).sum()

    # Calculate the daily range (high - low) and its ratio to the closing price
    df['daily_range'] = df['high'] - df['low']
    df['range_to_close_ratio'] = df['daily_range'] / df['close']

    # Evaluate the percentage of days where the range exceeds a certain threshold
    df['large_range'] = (df['daily_range'] / df['close']) > 0.05

    # Measure the difference between the opening price of the current day and the closing price of the previous day
    df['opening_gap'] = df['open'] - df['close'].shift(1)
    df['bullish_gap'] = (df['opening_gap'] > 0) & (df['close'] > df['open'])
    df['bearish_gap'] = (df['opening_gap'] < 0) & (df['close'] < df['open'])

    # Calculate the true range
    df['true_range'] = df[['high' - 'low', 'high' - 'close'].shift(1), 'close'.shift(1) - 'low']].max(axis=1)

    # Calculate the sum of true ranges over the past 14 days
    df['sum_true_range_14'] = df['true_range'].rolling(window=14).sum()

    # Calculate the 14-period ATR based on the true range
    df['ATR_14'] = df['true_range'].rolling(window=14).mean()

    # Combine factors into a composite factor
    df['composite_factor'] = (
        0.3 * df['EMA_diff'] + 
        0.2 * df['EMA_50_ROC'] + 
        0.1 * (df['vol_price_corr_30'] - df['up_down_ratio']) + 
        0.1 * df['range_to_close_ratio'] + 
        0.1 * df['opening_gap'] + 
        0.2 * df['ATR_14']
    )

    return df['composite_factor'].dropna()
