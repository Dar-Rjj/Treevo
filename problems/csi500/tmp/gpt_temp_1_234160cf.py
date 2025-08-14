import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    df['log_return_close'] = np.log(df['close'] / df['close'].shift(1))
    df['price_momentum'] = df['log_return_close'].rolling(window=5).sum()

    # Apply Combined Volume and Amount Shock Filter
    df['abs_vol_change'] = abs(df['volume'] - df['volume'].shift(1))
    df['abs_amount_change'] = abs(df['amount'] - df['amount'].shift(1))
    vol_threshold = df['abs_vol_change'].quantile(0.75)
    amount_threshold = df['abs_amount_change'].quantile(0.75)
    df = df[(df['abs_vol_change'] < vol_threshold) & (df['abs_amount_change'] < amount_threshold)]

    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']

    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['close'] - df['open']

    # Calculate Volume-Weighted Moving Average of Intraday Volatility
    df['vol_weighted_volatility'] = df['intraday_volatility'] * df['volume']
    df['vol_weighted_ma'] = df['vol_weighted_volatility'].rolling(window=10).mean() / df['volume'].rolling(window=10).mean()

    # Calculate Intraday Reversal Score
    df['reversal_score'] = df['close_to_open_return'] / df['vol_weighted_ma']

    # Apply Exponential Decay to Reversal Score
    half_life = 5
    df['reversal_score_decay'] = df['reversal_score'] * np.exp(-np.log(2) * (range(len(df)) / half_life))

    # Calculate Intraday Range
    df['intraday_range'] = (df['high'] - df['low']) / df['open']

    # Combine Reversal Score and Intraday Range
    df['combined_score'] = df['reversal_score_decay'] + df['intraday_range']
    df['combined_score_decay'] = df['combined_score'] * np.exp(-np.log(2) * (range(len(df)) / half_life))

    # Calculate Volume Momentum
    df['log_return_volume'] = np.log(df['volume'] / df['volume'].shift(1))
    df['volume_momentum'] = df['log_return_volume'].rolling(window=5).sum()

    # Confirm with Volume
    df['volume_change_pct'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)

    # Weighted Final Alpha Factor
    df['final_alpha_factor'] = df['price_momentum'] * df['volume_change_pct'] * df['volume_momentum']

    return df['final_alpha_factor']
