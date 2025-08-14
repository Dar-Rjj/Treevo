import pandas as pd
import pandas as pd

def heuristics_v2(df, w1=0.5, w2=0.5):
    # Calculate Simple Moving Averages (SMA)
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_100'] = df['close'].rolling(window=100).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()

    # Subtract Longer SMA from Shorter SMA
    df['SMA_diff_100_200'] = df['SMA_100'] - df['SMA_200']
    df['SMA_diff_50_100'] = df['SMA_50'] - df['SMA_100']

    # Calculate Daily Price Return
    df['daily_return'] = df['close'] / df['close'].shift(1) - 1

    # Calculate 20-Day Weighted Moving Average of Returns
    df['weighted_return'] = ((df['daily_return'] * df['volume']).rolling(window=20).sum()) / df['volume'].rolling(window=20).sum()

    # Adjust for Price Volatility
    df['avg_daily_range'] = (df['high'] - df['low']).rolling(window=22).mean()
    df['adjusted_weighted_return'] = df['weighted_return'] - df['avg_daily_range']

    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']

    # Determine Reversal Signal
    df['reversal_signal'] = df['intraday_return'].apply(lambda x: -1 if x > 0 else 1)

    # Multiply Subtracted SMA by Reversal Signal
    df['SMA_diff_100_200_reversed'] = df['SMA_diff_100_200'] * df['reversal_signal']
    df['SMA_diff_50_100_reversed'] = df['SMA_diff_50_100'] * df['reversal_signal']

    # Compute Average True Range (ATR)
    df['tr'] = df[['high' - 'low', 'high' - df['close'].shift(1), df['close'].shift(1) - df['low']]].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Multiply ATR by Combined Signal
    df['atr_combined_signal'] = df['atr'] * (df['SMA_diff_100_200_reversed'] + df['SMA_diff_50_100_reversed'])

    # Smooth Result with Exponential Moving Average (EMA)
    df['ema_atr_combined'] = df['atr_combined_signal'].ewm(span=20, adjust=False).mean()

    # High-Low Range Momentum with Volume Adjustment
    df['high_low_range'] = df['high'] - df['low']
    df['range_momentum'] = df['high_low_range'] - df['high_low_range'].shift(1)
    df['avg_volume'] = df['volume'].rolling(window=30).mean()
    df['range_momentum_adj'] = df['range_momentum'] / df['avg_volume']

    # Final Alpha Factor Combination
    combined_price_intraday = df['SMA_diff_50_100'] + df['SMA_diff_100_200'] + df['ema_atr_combined']
    final_alpha = w1 * combined_price_intraday + w2 * (df['range_momentum_adj'] + df['adjusted_weighted_return'])

    return final_alpha
