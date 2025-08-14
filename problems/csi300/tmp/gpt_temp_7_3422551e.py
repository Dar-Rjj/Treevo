import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Simple Moving Averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_60'] = df['close'].rolling(window=60).mean()

    # Analyze the difference and ratio between various SMAs
    df['SMA_diff_short_medium'] = df['SMA_5'] - df['SMA_20']
    df['SMA_diff_medium_long'] = df['SMA_20'] - df['SMA_60']
    df['SMA_ratio_short_long'] = df['SMA_5'] / df['SMA_60']

    # Examine volume in relation to price movements
    df['price_change'] = df['close'].pct_change()
    df['volume_weighted_positive'] = df['volume'] * (df['price_change'] > 0).astype(int)
    df['volume_weighted_negative'] = df['volume'] * (df['price_change'] < 0).astype(int)

    # Incorporate range (High - Low) for volatility insight
    df['range'] = df['high'] - df['low']
    df['range_ratio'] = df['range'] / df['close']

    # Average True Range (ATR) over a certain period (e.g., 14 days)
    df['tr'] = df[['high' - 'low', (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()]].max(axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()

    # Evaluate the relationship between open and close prices
    df['open_close_diff'] = df['open'] - df['close']
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['close'] <= df['open']).astype(int)

    # Investigate high and low prices to detect support and resistance levels
    df['high_low_diff'] = df['high'] - df['low']
    df['touch_high'] = (df['high'] == df['high'].rolling(window=20).max()).astype(int)
    df['touch_low'] = (df['low'] == df['low'].rolling(window=20).min()).astype(int)

    # Integrate trade amount data
    df['amount_weighted_price_change'] = df['amount'] * df['price_change']
    df['cumulative_amount_20'] = df['amount'].rolling(window=20).sum()

    # Logarithmic Return
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))

    # Volume-Weighted Daily Return
    df['vol_weighted_return'] = df['log_return'] * df['volume']

    # Dynamic Price Volatility
    df['abs_log_return'] = df['log_return'].abs()
    df['ema_volatility'] = df['abs_log_return'].ewm(span=20, adjust=False).mean()

    # Adjust Volume-Weighted Return for Volatility
    df['adjusted_vol_weighted_return'] = df['vol_weighted_return'] / df['ema_volatility']

    # Cumulate Adjusted Volume-Weighted Values Over Window
    df['cumulated_adjusted_vol_weighted_return'] = df['adjusted_vol_weighted_return'].rolling(window=20).sum()

    # Add Enhanced Trend Following Component
    df['sma_high_low_20'] = (df['high'] + df['low']) / 2.0
    df['enhanced_trend_signal'] = np.where(df['close'] > df['sma_high_low_20'], 1.5, -0.5)

    # Combine Adjusted Volume-Weighted Returns with Enhanced Trend Signal
    df['final_metric'] = df['cumulated_adjusted_vol_weighted_return'] * df['enhanced_trend_signal']

    return df['final_metric']
