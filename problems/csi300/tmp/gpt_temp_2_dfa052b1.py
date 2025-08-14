import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate EMAs
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # EMA crossover factor
    df['ema_crossover'] = 0
    df.loc[df['ema_5'] > df['ema_20'], 'ema_crossover'] = 1
    df.loc[df['ema_5'] < df['ema_20'], 'ema_crossover'] = -1

    # Price rate of change over 10 days
    df['price_roc'] = df['close'].pct_change(periods=10)
    df['smoothed_price_roc'] = df['price_roc'].rolling(window=10).mean()
    df['roc_factor'] = np.sign(df['smoothed_price_roc'])

    # Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['vwap_factor'] = np.sign(df['close'] - df['vwap'])

    # Volume trend
    df['volume_diff'] = df['volume'].diff()
    df['volume_trend'] = df['volume_diff'].apply(np.sign).rolling(window=10).sum()

    # Amount-to-volume ratio
    df['amount_to_volume_ratio'] = df['amount'] / df['volume']
    df['amount_to_volume_factor'] = np.sign(df['amount_to_volume_ratio'] - df['amount_to_volume_ratio'].rolling(window=20).mean())

    # Relative Strength (RS) with dynamic lookback window
    def rs_factor(row):
        if row['volatility'] > 0.05:
            return row['close'] / row['close_10']
        else:
            return row['close'] / row['close_20']
    
    df['close_10'] = df['close'].shift(10)
    df['close_20'] = df['close'].shift(20)
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['rs'] = df.apply(rs_factor, axis=1)

    # Daily price range
    df['range'] = df['high'] - df['low']
    df['average_range'] = df['range'].rolling(window=20).mean()
    df['range_factor'] = np.where(df['range'] > 1.5 * df['average_range'], 1, -1)

    # True Range
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['avg_true_range'] = df['true_range'].rolling(window=20).mean()
    df['tr_factor'] = df['true_range'] / df['avg_true_range']

    # Combine factors
    df['combined_factor'] = df['ema_crossover'] + df['roc_factor'] + df['vwap_factor'] + df['volume_trend'] + df['amount_to_volume_factor'] + df['rs'] + df['range_factor'] + df['tr_factor']

    # Final alpha factor
    alpha_factor = df['combined_factor']

    return alpha_factor
