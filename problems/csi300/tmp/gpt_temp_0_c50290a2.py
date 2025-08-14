import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Moving Averages
    df['5_day_SMA'] = df['close'].rolling(window=5).mean()
    df['20_day_SMA'] = df['close'].rolling(window=20).mean()
    df['5_day_EMA'] = df['close'].ewm(span=5, adjust=False).mean()
    df['20_day_EMA'] = df['close'].ewm(span=20, adjust=False).mean()

    # Calculate Momentum Indicators
    df['10_day_momentum'] = df['close'] - df['close'].shift(10)
    df['30_day_momentum'] = df['close'] - df['close'].shift(30)

    # Calculate Volatility Indicators
    df['daily_returns'] = (df['close'] - df['open']) / df['open']
    df['10_day_std'] = df['daily_returns'].rolling(window=10).std()
    df['30_day_std'] = df['daily_returns'].rolling(window=30).std()

    # Calculate Daily Range
    df['daily_range'] = df['high'] - df['low']

    # Calculate Average True Range
    df['true_range'] = df[['high' - 'low', abs('high' - df['close'].shift(1)), abs('low' - df['close'].shift(1))]].max(axis=1)
    df['14_day_ATR'] = df['true_range'].rolling(window=14).mean()

    # Calculate Volume Moving Averages
    df['5_day_vol_MA'] = df['volume'].rolling(window=5).mean()
    df['20_day_vol_MA'] = df['volume'].rolling(window=20).mean()

    # Calculate Daily Return
    df['daily_return'] = (df['close'] - df['open']) / df['open']

    # Calculate Overlap Studies: Bollinger Bands
    df['20_day_SMA_close'] = df['close'].rolling(window=20).mean()
    df['20_day_std_close'] = df['close'].rolling(window=20).std()
    df['upper_bband'] = df['20_day_SMA_close'] + 2 * df['20_day_std_close']
    df['lower_bband'] = df['20_day_SMA_close'] - 2 * df['20_day_std_close']

    # Calculate Overlap Studies: Keltner Channels
    df['20_day_EMA_close'] = df['close'].ewm(span=20, adjust=False).mean()
    df['10_day_ATR'] = df['true_range'].rolling(window=10).mean()
    df['upper_keltner'] = df['20_day_EMA_close'] + 2 * df['10_day_ATR']
    df['lower_keltner'] = df['20_day_EMA_close'] - 2 * df['10_day_ATR']

    # Combine Multiple Indicators: Composite Momentum
    df['composite_momentum'] = df['10_day_momentum'] + df['30_day_momentum']

    # Combine Multiple Indicators: Composite Volatility
    df['composite_volatility'] = df['10_day_std'] + df['30_day_std']

    # Final Alpha Factor
    alpha_factor = (
        df['composite_momentum'] / df['composite_volatility'] +
        (df['5_day_SMA'] - df['20_day_SMA']) / df['close'] +
        (df['5_day_EMA'] - df['20_day_EMA']) / df['close'] +
        (df['upper_bband'] - df['lower_bband']) / df['close'] +
        (df['upper_keltner'] - df['lower_keltner']) / df['close']
    )

    return alpha_factor
