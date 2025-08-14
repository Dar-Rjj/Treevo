import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Return
    df['daily_return'] = df['close'].pct_change()

    # Calculate 20-Day Volume-Weighted Average Return
    df['weighted_return'] = df['daily_return'] * df['volume']
    df['sum_weighted_return_20d'] = df['weighted_return'].rolling(window=20).sum()
    df['total_volume_20d'] = df['volume'].rolling(window=20).sum()
    df['volume_weighted_avg_return_20d'] = df['sum_weighted_return_20d'] / df['total_volume_20d']

    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']

    # Calculate Reversal Signal
    df['reversal_signal'] = df['intraday_return'].apply(lambda x: -1 if x > 0 else 1)

    # Calculate Intraday Return Ratio
    df['intraday_return_ratio'] = (df['high'] - df['low']) / df['open']

    # Calculate Weighted Open-to-Close Return
    df['open_to_close_return'] = (df['close'] - df['open']) / df['open']
    df['weighted_open_to_close_return'] = df['open_to_close_return'] * df['volume']

    # Adjust for Price Volatility
    df['price_range'] = df['high'] - df['low']
    df['avg_price_range_20d'] = df['price_range'].rolling(window=20).mean()

    # Final Factor Value
    df['momentum_volatility_adjusted'] = df['volume_weighted_avg_return_20d'] - df['avg_price_range_20d']

    # Combine Adjusted Momentum, Intraday, and Volatility Factors
    df['combined_factor'] = df['volume_weighted_avg_return_20d'] + df['intraday_return_ratio'] + df['weighted_open_to_close_return'] - df['avg_price_range_20d']

    # Price Momentum Component
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['sma_diff_50_100'] = df['sma_50'] - df['sma_100']
    df['sma_diff_100_200'] = df['sma_100'] - df['sma_200']
    df['sma_combined'] = df['sma_diff_50_100'] + df['sma_diff_100_200']

    # Intraday Reversal Momentum Component
    df['atr'] = df[['high', 'low', df['close'].shift(1)]].max(axis=1) - df[['high', 'low', df['close'].shift(1)]].min(axis=1)
    df['reversal_atr'] = df['atr'] * df['reversal_signal']
    df['ema_reversal_atr'] = df['reversal_atr'].ewm(span=10, adjust=False).mean()

    # Final Alpha Factor
    df['alpha_factor'] = df['combined_factor'] + df['sma_combined'] + df['ema_reversal_atr']
    
    return df['alpha_factor']
