import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Returns
    df['daily_return'] = df['close'].pct_change()

    # Calculate 10-day Moving Average of Returns
    df['ma_10d_return'] = df['daily_return'].rolling(window=10).mean()

    # Calculate Price Volatility
    df['price_volatility'] = (df['high'] - df['low']) / df['close']

    # Calculate Volume Spike Factor
    df['volume_ma_20d'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = (df['volume'] > df['volume_ma_20d']).astype(int)
    df['adjusted_ma_10d_return'] = df['ma_10d_return'] * (1 + df['volume_spike'])

    # Adjust 10-day MA for High Price Volatility
    df['high_price_volatility'] = (df['price_volatility'] > df['price_volatility'].quantile(0.75)).astype(int)
    df['final_ma_10d_return'] = df['adjusted_ma_10d_return'] * (1 + df['high_price_volatility'])

    # Calculate Intraday and High-Low Movements
    df['intraday_high_low_spread'] = df['high'] - df['low']
    df['intraday_close_open_change'] = df['close'] - df['open']
    df['sum_intraday_movements'] = df['intraday_high_low_spread'] + df['intraday_close_open_change']
    df['weighted_intraday_movements'] = df['sum_intraday_movements'] * df['volume']

    # High-Low Range Momentum
    df['current_high_low_range'] = df['high'] - df['low']
    df['previous_high_low_range'] = df['high'].shift(1) - df['low'].shift(1)
    df['high_low_momentum'] = df['current_high_low_range'] - df['previous_high_low_range']

    # Combine Components
    df['intraday_and_high_low_momentum'] = df['weighted_intraday_movements'] + df['high_low_momentum']

    # Adjust for Volume-Weighted Return
    df['volume_weighted_return'] = df['daily_return'] * df['volume']
    df['factor'] = df['final_ma_10d_return'] + df['intraday_and_high_low_momentum']
    df['factor'] = df.apply(lambda row: row['factor'] * row['volume_weighted_return'] if row['volume_weighted_return'] > 0 else row['factor'] / abs(row['volume_weighted_return']), axis=1)

    return df['factor']
