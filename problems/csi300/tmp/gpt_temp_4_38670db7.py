import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 21-Day Volume-Weighted Intraday Return
    df['daily_intraday_return'] = df['close'] - df['open']
    df['volume_weighted_intraday_return'] = (df['daily_intraday_return'] * df['volume']).rolling(window=21).sum() / df['volume'].rolling(window=21).sum()

    # Calculate Intraday Reversal
    df['intraday_high_low_spread'] = df['high'] - df['low']
    df['close_to_open_return'] = df['close'] - df['open']
    df['intraday_reversal'] = (df['close_to_open_return'] / df['intraday_high_low_spread']) * df['intraday_high_low_spread']

    # Enhance with Volume-Weighted High-Low Difference
    df['high_low_diff'] = df['high'] - df['low']
    df['volume_weighted_high_low_diff'] = (df['high_low_diff'] * df['volume']).rolling(window=21).sum() / df['volume'].rolling(window=21).sum()

    # Incorporate Volume and Amount Influence
    df['avg_volume'] = df['volume'].rolling(window=21).mean()
    df['intraday_volume_impact'] = df['volume'] / df['avg_volume']
    df['amount_impact'] = df['amount'] / df['avg_volume']
    df['combined_volume_amount_impact'] = df['intraday_volume_impact'] + df['amount_impact']

    # Adjust for Volatility
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_volatility'] = df['log_returns'].rolling(window=21).std()
    df['adjusted_intraday_return'] = df['volume_weighted_intraday_return'] / df['realized_volatility']
    df['adjusted_intraday_reversal'] = df['intraday_reversal'] / df['realized_volatility']
    df['adjusted_volume_weighted_high_low_diff'] = df['volume_weighted_high_low_diff'] / df['realized_volatility']

    # Consider Sign of Close-to-Open Return
    df['close_to_open_sign'] = np.sign(df['close'] - df['open'])

    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high_low_diff'].rolling(window=21).sum()

    # Calculate Close-to-Close Return
    df['close_to_close_return'] = df['close'] - df['close'].shift(1)

    # Adjust Return by Intraday Volatility and Volume-Weighted Factor
    df['adjusted_return'] = (df['close_to_close_return'] * df['adjusted_intraday_volatility'] * df['adjusted_volume_weighted_high_low_diff'])

    # Introduce Lagged Close-to-Open Return
    df['lagged_close_to_open_return'] = df['close'].shift(1) - df['open'].shift(1)
    df['lagged_close_to_open_sign'] = np.sign(df['lagged_close_to_open_return'])

    # Adjust for Lagged Momentum
    df['lagged_momentum'] = df['lagged_close_to_open_return'] * df['close_to_close_return']

    # Integrate Price Range Ratio
    df['intraday_price_range'] = df['high'] - df['low']
    df['long_term_price_range'] = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    df['price_range_ratio'] = df['intraday_price_range'] / df['long_term_price_range']

    # Adjust for Price Range Ratio
    df['adjusted_lagged_momentum'] = df['lagged_momentum'] * df['price_range_ratio']

    # Introduce Intraday Momentum
    df['7_day_avg_intraday_return'] = df['daily_intraday_return'].rolling(window=7).mean()
    df['intraday_momentum'] = df['7_day_avg_intraday_return'] * df['intraday_reversal']

    # Integrate Intraday Momentum
    df['final_adjusted_return'] = df['adjusted_lagged_momentum'] + df['intraday_momentum']

    # Factor Value
    factor_value = df['final_adjusted_return']

    return factor_value
