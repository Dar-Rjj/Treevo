import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return_high'] = (df['high'] - df['open']) / df['open']
    df['intraday_return_low'] = (df['low'] - df['open']) / df['open']
    df['intraday_return'] = (df['close'] - df['open']) / df['open']

    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']

    # Calculate Intraday Volatility (Average True Range over a period)
    df['true_range'] = df[['high' - 'low', abs('high' - df['close'].shift(1)), abs('low' - df['close'].shift(1))]].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()

    # Calculate Volume Trend
    df['volume_ema'] = df['volume'].ewm(span=14, adjust=False).mean()

    # Compute Volume Reversal Component
    df['volume_reversal_sentiment'] = np.where(df['volume'] > df['volume_ema'], 1, -1)
    df['volume_reversal_component'] = df['intraday_return'] * df['volume_reversal_sentiment']

    # Enhance Volume Reversal Component by Intraday Momentum
    df['volume_reversal_adjustment'] = np.sign(df['intraday_return'] * df['intraday_range'])
    df['enhanced_volume_reversal'] = df['volume_reversal_component'] * df['volume_reversal_adjustment']

    # Calculate Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['high'] - df['low']

    # Calculate Intraday Momentum
    df['intraday_momentum'] = df['intraday_return'] * df['intraday_range'] - df['intraday_high_low_spread'].shift(1)

    # Calculate Intraday Reversal
    df['intraday_reversal'] = (df['close'] - df['open']) - (df['close'].shift(1) - df['open'].shift(1))
    df['intraday_reversal'] *= df['intraday_momentum']

    # Calculate Price Momentum
    df['price_return_1_period'] = df['close'] / df['close'].shift(1) - 1
    df['price_momentum'] = df['price_return_1_period'].rolling(window=5).mean()

    # Generate Initial Alpha Factor
    df['initial_alpha_factor'] = df['intraday_high_low_spread'] * df['price_momentum']

    # Confirm with Volume
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_confirmation'] = df['initial_alpha_factor'] * np.log(df['volume_change'] + 1)

    # Final Alpha Factor
    df['final_alpha_factor'] = df['volume_confirmation'] * np.sign(df['volume_change'])

    return df['final_alpha_factor']
