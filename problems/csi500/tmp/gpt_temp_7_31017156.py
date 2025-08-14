import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Returns
    df['daily_return'] = df['close'].pct_change()

    # Calculate 10-day Moving Average of Returns
    df['10_day_ma_returns'] = df['daily_return'].rolling(window=10).mean()

    # Calculate Price Volatility
    df['price_volatility'] = (df['high'] - df['low']) / df['close']

    # Calculate Volume Spike Factor
    df['volume_20_day_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > df['volume_20_day_ma']
    df['adjusted_10_day_ma'] = df.apply(
        lambda row: row['10_day_ma_returns'] * 1.1 if row['volume_spike'] else row['10_day_ma_returns'], axis=1
    )

    # Calculate High Price Volatility Factor
    high_price_volatility = df['price_volatility'] > df['price_volatility'].rolling(window=10).quantile(0.75)
    df['adjusted_10_day_ma'] = df.apply(
        lambda row: row['adjusted_10_day_ma'] * 1.2 if high_price_volatility.loc[row.name] else row['adjusted_10_day_ma'], axis=1
    )

    # Calculate Amount Change
    df['amount_change'] = df['amount'].pct_change()

    # Adjust for Sudden Amount Increase
    sudden_amount_increase = df['amount_change'] > 1.5
    df['adjusted_10_day_ma'] = df.apply(
        lambda row: row['adjusted_10_day_ma'] * 1.3 if sudden_amount_increase.loc[row.name] else row['adjusted_10_day_ma'], axis=1
    )

    # Calculate Intraday and High-Low Movements
    df['intraday_high_low_spread'] = df['high'] - df['low']
    df['intraday_close_open_change'] = df['close'] - df['open']
    df['sum_intraday_movements'] = df['intraday_high_low_spread'] + df['intraday_close_open_change']
    df['weighted_intraday_movements'] = df['sum_intraday_movements'] * df['volume']

    # High-Low Range Momentum
    df['current_high_low_range'] = df['high'] - df['low']
    df['previous_high_low_range'] = df['high'].shift(1) - df['low'].shift(1)
    df['high_low_range_momentum'] = df['current_high_low_range'] - df['previous_high_low_range']

    # Volume-Weighted Return
    df['volume_weighted_return'] = df['daily_return'] * df['volume']

    # Combine Components
    df['weighted_intraday_and_high_low_momentum'] = df['weighted_intraday_movements'] + df['high_low_range_momentum']
    df['final_factor'] = df['adjusted_10_day_ma']
    df['final_factor'] = df.apply(
        lambda row: row['final_factor'] * row['volume_weighted_return'] if row['volume_weighted_return'] > 0 else row['final_factor'] / abs(row['volume_weighted_return']), axis=1
    )
    df['final_factor'] += df['weighted_intraday_and_high_low_momentum']

    return df['final_factor']
