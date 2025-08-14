import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Return
    df['daily_return'] = df['close'].pct_change()

    # Calculate 20-Day Weighted Moving Average of Returns
    df['weighted_return'] = df['daily_return'] * df['volume']
    df['sum_weighted_return'] = df['weighted_return'].rolling(window=20).sum()
    df['sum_volume'] = df['volume'].rolling(window=20).sum()
    df['weighted_moving_avg_return'] = df['sum_weighted_return'] / df['sum_volume']

    # Adjust for Price Volatility
    df['price_range'] = df['high'] - df['low']
    df['avg_price_range'] = df['price_range'].rolling(window=22).mean()
    df['volatility_adjusted_return'] = df['weighted_moving_avg_return'] - df['avg_price_range']

    # Calculate Intraday Return Ratio
    df['intraday_return_ratio'] = (df['high'] - df['low']) / (df['high'] + df['low'])

    # Calculate Weighted Open-to-Close Return
    df['open_to_close_return'] = (df['close'] - df['open']) * df['volume']

    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']

    # Calculate High-Low Range Momentum
    df['high_low_range_momentum'] = df['high_low_range'].diff()

    # Adjust by Volume
    df['avg_volume'] = df['volume'].rolling(window=20).mean()

    # Calculate Price Trend
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['price_trend'] = df['sma_50'] - df['close']

    # Combine Adjusted Momentum, Intraday, and High-Low Range Factors
    df['combined_factors'] = (
        df['volatility_adjusted_return'] + 
        df['intraday_return_ratio'] + 
        df['open_to_close_return'] + 
        df['high_low_range_momentum']
    )

    # Price Momentum Component
    df['price_momentum'] = (df['sma_50'] - df['sma_100']) + (df['sma_100'] - df['sma_200'])

    # Intraday Reversal Momentum Component
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['reversal_signal'] = df['intraday_return'].apply(lambda x: -1 if x > 0 else 1)
    df['atr'] = df[['high', 'low', df['close'].shift(1)]].max(axis=1) - df[['high', 'low', df['close'].shift(1)]].min(axis=1)
    df['intraday_reversal_momentum'] = df['atr'] * df['reversal_signal']

    # Final Alpha Factor
    df['alpha_factor'] = (
        df['combined_factors'] + 
        df['price_momentum'] + 
        df['intraday_reversal_momentum']
    ) / df['avg_volume'] * df['price_trend']

    return df['alpha_factor']
