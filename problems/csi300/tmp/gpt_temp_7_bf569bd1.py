import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Reversal: 2 * (High - Low) / (Close + Open)
    df['intraday_reversal'] = 2 * (df['high'] - df['low']) / (df['close'] + df['open'])

    # Adjust for Momentum and Volume Change
    df['momentum_adjustment'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['adjusted_intraday_reversal'] = (1 + df['momentum_adjustment']) * (1 + df['volume_change'])

    # Compute Intraday Midpoint and Close to Midpoint Deviation
    df['midpoint'] = (df['high'] + df['low']) / 2
    df['close_to_midpoint_deviation'] = df['close'] - df['midpoint']

    # Generate Day-to-Day Open Price Change
    df['open_price_change'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    # Combine Intraday Reversal, Close to Midpoint Deviation, and Open Price Change
    df['combined_factor_1'] = df['adjusted_intraday_reversal'] * df['close_to_midpoint_deviation'] + df['open_price_change']

    # Calculate Daily Volume Surprise
    df['vol_moving_avg'] = df['volume'].rolling(window=10).mean()
    df['volume_surprise'] = df['volume'] - df['vol_moving_avg']

    # Compute Volume Influence Ratio
    df['upward_volume'] = df.apply(lambda row: row['volume'] if row['close'] > row['open'] else 0, axis=1).rolling(window=10).sum()
    df['downward_volume'] = df.apply(lambda row: row['volume'] if row['close'] < row['open'] else 0, axis=1).rolling(window=10).sum()
    df['volume_influence_ratio'] = df['upward_volume'] / df['downward_volume']

    # Calculate Smoothed Price Momentum
    df['price_momentum'] = df['close'] - df['close'].shift(1)
    df['smoothed_price_momentum'] = df['price_momentum'].ewm(span=10, adjust=False).mean()

    # Adjust by Smoothed Volume Change
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['smoothed_volume_ratio'] = df['volume_ratio'].rolling(window=10).mean()
    df['combined_momentum_and_volume'] = df['smoothed_price_momentum'] * df['smoothed_volume_ratio']

    # Compute Intraday Close-Open Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['intraday_return_5d_ma'] = df['intraday_return'].rolling(window=5).mean()

    # Calculate 5-Day Moving Average of Intraday Close-Open Return
    df['intraday_momentum'] = (df['high'] - df['low']) * df['intraday_return']

    # Integrate Intraday Components
    df['integrated_intraday_factor'] = (df['intraday_momentum'] * df['intraday_return_5d_ma']) * df['volume_surprise']

    # Final Alpha Factor
    df['final_alpha_factor'] = (df['combined_factor_1'] * df['integrated_intraday_factor']) * df['volume_influence_ratio'] + df['adjusted_intraday_reversal'] + df['combined_momentum_and_volume'] - (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    return df['final_alpha_factor']
