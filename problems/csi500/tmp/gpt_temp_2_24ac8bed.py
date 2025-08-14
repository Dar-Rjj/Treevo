import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Combined Price Momentum
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['price_momentum'] = df['log_return'].rolling(window=5).sum()

    # Apply Volume Shock Filter
    df['abs_volume_change'] = np.abs(df['volume'] - df['volume'].shift(1))
    volume_threshold = df['abs_volume_change'].quantile(0.80)
    df = df[df['abs_volume_change'] < volume_threshold]

    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']

    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['close'] - df['open']

    # Calculate Volume-Weighted Moving Average of Intraday Volatility
    df['vol_weighted_intraday_volatility'] = (df['intraday_volatility'] * df['volume']).rolling(window=15).mean()

    # Calculate Intraday Reversal Score
    df['reversal_score'] = df['close_to_open_return'] / df['vol_weighted_intraday_volatility']

    # Apply Exponential Decay to Reversal Score
    half_life = 7
    df['decayed_reversal_score'] = df['reversal_score'].ewm(half_life=half_life, adjust=False).mean()

    # Calculate Intraday Range
    df['intraday_range'] = (df['high'] - df['low']) / df['open']

    # Calculate Volume-Weighted Moving Average of Intraday Range
    df['vol_weighted_intraday_range'] = (df['intraday_range'] * df['volume']).rolling(window=15).mean()

    # Combine Reversal Score and Intraday Range
    df['combined_reversal_range'] = df['decayed_reversal_score'] + df['vol_weighted_intraday_range']
    df['decayed_combined_reversal_range'] = df['combined_reversal_range'].ewm(half_life=half_life, adjust=False).mean()

    # Calculate Intraday Price Movement
    df['price_movement_1'] = (df['high'] - df['low']) / df['open']
    df['price_movement_2'] = (df['close'] - df['open']) / df['open']
    df['avg_price_movement'] = (df['price_movement_1'] + df['price_movement_2']) / 2

    # Smooth the Average using a 5-day Exponential Moving Average
    df['smoothed_price_movement'] = df['avg_price_movement'].ewm(span=5, adjust=False).mean()

    # Confirm with Volume
    df['volume_change_pct'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)

    # Weighted Final Alpha Factor
    df['weighted_alpha_factor'] = df['smoothed_price_movement'] * df['volume_change_pct']

    # Calculate Close to High and Low Ratios
    df['close_high_ratio'] = df['close'] / df['high']
    df['close_low_ratio'] = df['close'] / df['low']

    # Combine Close to High and Low Ratios
    df['combined_ratio'] = 0.6 * df['close_high_ratio'] + 0.4 * df['close_low_ratio']

    # Final Alpha Factor
    df['final_alpha_factor'] = df['weighted_alpha_factor'] * df['combined_ratio']

    return df['final_alpha_factor']
