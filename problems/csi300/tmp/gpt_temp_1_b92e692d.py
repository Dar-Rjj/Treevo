import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Change
    df['price_change'] = df['close'].diff()

    # Compute 10-Day Moving Average of Daily Price Changes
    df['ma_10_price_change'] = df['price_change'].rolling(window=10).mean()

    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['low']

    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']

    # Combine Intraday and Close-to-Open Returns with weights by absolute value
    df['combined_return'] = (abs(df['intraday_return']) * df['intraday_return'] + 
                             abs(df['close_to_open_return']) * df['close_to_open_return']) / \
                            (abs(df['intraday_return']) + abs(df['close_to_open_return']))

    # Integrate Volume Change
    df['volume_change'] = df['volume'].diff()
    df['weighted_combined_return'] = df['combined_return'] * df['volume_change']
    df['preliminary_factor'] = df['weighted_combined_return'].rolling(window=10).sum()

    # Calculate True Range (TR)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    # Calculate Average True Range (ATR)
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Calculate Positive Directional Movement (+DM)
    df['+dm'] = df.apply(lambda row: max(row['high'] - df['high'].shift(1)[row.name], 0) if row['high'] > df['high'].shift(1)[row.name] and row['low'] >= df['low'].shift(1)[row.name] else 0, axis=1)

    # Calculate Negative Directional Movement (-DM)
    df['-dm'] = df.apply(lambda row: max(df['low'].shift(1)[row.name] - row['low'], 0) if row['low'] < df['low'].shift(1)[row.name] and row['high'] <= df['high'].shift(1)[row.name] else 0, axis=1)

    # Calculate +DM Smoothed Over 14 Periods
    df['+dm_smoothed'] = df['+dm'].rolling(window=14).sum() / 14

    # Calculate -DM Smoothed Over 14 Periods
    df['-dm_smoothed'] = df['-dm'].rolling(window=14).sum() / 14

    # Calculate +DI
    df['+di'] = df['+dm_smoothed'] / df['atr']

    # Calculate -DI
    df['-di'] = df['-dm_smoothed'] / df['atr']

    # Calculate ADMI
    df['admi'] = (df['+di'] - df['-di']) / (df['+di'] + df['-di'])

    # Synthesize Final Alpha Factor
    df['diff_ma_preliminary'] = df['ma_10_price_change'] - df['preliminary_factor']
    df['final_alpha_factor'] = df['diff_ma_preliminary'].rolling(window=5).mean() * df['admi']

    return df['final_alpha_factor']
