import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Momentum
    high_low_range = df['high'] - df['low']
    intraday_momentum = (df['high'] - df['low']) / df['open']

    # Calculate Volume Flow
    volume_difference = df['volume'].diff()
    average_volume = (df['volume'] + df['volume'].shift(1)) / 2
    volume_flow = volume_difference / average_volume

    # Combine Intraday Momentum and Volume Flow
    combined_factor = intraday_momentum * volume_flow

    # Compute Intraday Volatility
    intraday_prices = df[['open', 'high', 'low', 'close']].T
    intraday_volatility = intraday_prices.std()

    # Weight by Intraday Volatility
    volatility_weighted_factor = combined_factor * intraday_volatility

    # Calculate Momentum Based on Close Prices
    close_delta = df['close'].diff()

    # Incorporate Volume into Momentum
    scaled_momentum = close_delta / (df['volume'] ** 0.5) * 100

    # Maintain a Rolling Window for Past N Days (e.g., 5 days)
    n_days = 5
    rolling_scaled_momentum = df['close'].rolling(window=n_days, min_periods=1).apply(
        lambda x: sum(scaled_momentum.iloc[i] for i in range(len(x))) / len(x), raw=False
    )

    # Incorporate Price Volatility
    close_prices = df['close']
    price_volatility = close_prices.rolling(window=n_days, min_periods=1).std()
    adjusted_scaled_momentum = rolling_scaled_momentum / price_volatility

    # Incorporate Price Range
    daily_price_range = df['high'] - df['low']
    adjusted_scaled_momentum_range = rolling_scaled_momentum / daily_price_range
    final_combined_momentum = (adjusted_scaled_momentum + adjusted_scaled_momentum_range) / 2

    # Calculate Short-Term Moving Average (5 days)
    short_term_ma = df['close'].rolling(window=5, min_periods=1).mean()

    # Calculate Long-Term Moving Average (20 days)
    long_term_ma = df['close'].rolling(window=20, min_periods=1).mean()

    # Subtract Short-Term from Long-Term MA
    ma_signal = short_term_ma - long_term_ma

    # Determine Momentum Signal
    momentum_signal = ma_signal.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    momentum_signal = momentum_signal * df['volume']

    # Combine Momentum and Moving Average Signals
    combined_signal = volatility_weighted_factor + final_combined_momentum + momentum_signal

    # Summarize Momentum Over Multiple Days
    cumulative_signal = combined_signal.rolling(window=5, min_periods=1).sum()

    return cumulative_signal
