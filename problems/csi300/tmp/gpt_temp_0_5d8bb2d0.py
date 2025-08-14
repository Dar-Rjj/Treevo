import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    intraday_range = df['high'] - df['low']
    intraday_momentum = intraday_range / df['open']

    # Calculate Volume Flow
    volume_difference = df['volume'].diff()
    average_volume = (df['volume'] + df['volume'].shift(1)) / 2
    volume_flow = volume_difference / average_volume

    # Combine Intraday Momentum and Volume Flow
    combined_factor = intraday_momentum * volume_flow

    # Compute Intraday Volatility
    intraday_volatility = df[['open', 'high', 'low', 'close']].std(axis=1)

    # Weight by Intraday Volatility
    weighted_factor = combined_factor * intraday_volatility

    # Calculate Close Price Momentum
    close_momentum = df['close'].diff()

    # Incorporate Volume into Momentum
    scaled_momentum = close_movement / np.sqrt(df['volume']) * 10
    N = 5
    rolling_scaled_momentum = scaled_momentum.rolling(window=N).sum()

    # Incorporate Price Volatility
    price_volatility = df['close'].rolling(window=N).std()
    adjusted_scaled_momentum = rolling_scaled_momentum / price_volatility

    # Incorporate Price Range
    daily_price_range = df['high'] - df['low']
    final_scaled_momentum = adjusted_scaled_momentum * daily_price_range

    # Calculate Short-Term and Long-Term Moving Averages
    short_term_ma = df['close'].rolling(window=5).mean()
    long_term_ma = df['close'].rolling(window=20).mean()

    # Subtract Short-Term from Long-Term MA
    ma_signal = short_term_ma - long_term_ma

    # Determine Momentum Signal
    momentum_signal = np.where(ma_signal > 0, 1, np.where(ma_signal < 0, -1, 0))

    # Combine Momentum and Moving Average Signals
    combined_signal = final_scaled_momentum + (momentum_signal * df['volume'])

    # Summarize Momentum Over Multiple Days
    cumulative_signal = combined_signal.rolling(window=5).sum()

    return cumulative_signal
