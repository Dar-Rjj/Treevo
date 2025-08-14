import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term and Long-Term Average Returns
    short_term_ma = df['close'].rolling(window=5).mean()
    long_term_ma = df['close'].rolling(window=20).mean()
    avg_return_diff = long_term_ma - short_term_ma
    volume_adjusted_diff = avg_return_diff * df['volume']

    # Calculate Intraday Price Changes
    intraday_high_low_diff = df['high'] - df['low']
    intraday_open_close_diff = df['open'] - df['close']
    combined_intraday_diff = intraday_high_low_diff + intraday_open_close_diff
    close_std = df['close'].rolling(window=10).std()
    intraday_volatility = combined_intraday_diff * close_std

    # Calculate Volume-Weighted Intraday Momentum
    avg_intraday_price = (df['high'] + df['low'] + df['open'] + df['close']) / 4
    volume_weighted_momentum = avg_intraday_price * df['volume']

    # Compute Final Combined Momentum Factor
    combined_momentum = volume_adjusted_diff + volume_weighted_momentum + intraday_volatility
    final_momentum = combined_momentum.ewm(span=5, adjust=False).mean()

    # Integrate Trend Strength
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    final_momentum_with_trend = final_momentum * adx

    return final_momentum_with_trend
