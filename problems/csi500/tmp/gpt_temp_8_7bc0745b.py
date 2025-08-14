import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Define lookback periods
    short_lookback = 10
    long_lookback = 50
    price_change_lookback = 10
    turnover_lookback = 20
    vol_lookback = 20

    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA'] = df['close'].rolling(window=short_lookback).mean()
    df['SMA_long'] = df['close'].rolling(window=long_lookback).mean()

    # Compute Volume-Adjusted Volatility
    df['HL_diff'] = df['high'] - df['low']
    df['volume_weighted_HL_diff'] = df['HL_diff'] * df['volume']
    df['vol_adj_vol'] = df['volume_weighted_HL_diff'].rolling(window=vol_lookback).mean()

    # Compute Price Momentum
    df['price_momentum'] = (df['close'] - df['SMA']) / df['close'].rolling(window=price_change_lookback).mean()

    # Incorporate Additional Price Change Metrics
    df['pct_change_close'] = df['close'].pct_change(periods=price_change_lookback)
    df['hl_range'] = df['high'] - df['low']

    # Consider Market Trend Alignment
    df['trend_indicator'] = (df['SMA'] > df['SMA_long']).astype(int)

    # Incorporate Liquidity Measures
    df['daily_turnover'] = df['volume'] * df['close']
    df['avg_turnover'] = df['daily_turnover'].rolling(window=turnover_lookback).mean()

    # Adaptive Lookback Periods
    df['historical_vol'] = df['close'].pct_change().rolling(window=vol_lookback).std() * (252 ** 0.5)
    df['adaptive_short_lookback'] = df['historical_vol'].apply(lambda x: int(short_lookback * (1 + 0.1 * (x - 0.1))))
    df['adaptive_long_lookback'] = df['historical_vol'].apply(lambda x: int(long_lookback * (1 + 0.1 * (x - 0.1))))

    # Dynamic Weight Adjustments
    df['weight_price_momentum'] = df['trend_indicator'] * 0.6 + (1 - df['trend_indicator']) * 0.4
    df['weight_vol_adj_vol'] = 1 - df['weight_price_momentum']
    df['weight_liquidity'] = df['avg_turnover'].rank(pct=True)
    df['weight_trend'] = df['trend_indicator'] * 0.7 + (1 - df['trend_indicator']) * 0.3

    # Final Alpha Factor
    df['alpha_factor'] = (df['price_momentum'] * df['weight_price_momentum'] +
                           df['vol_adj_vol'] * df['weight_vol_adj_vol'] +
                           df['pct_change_close'] * 0.1 +
                           df['hl_range'] * 0.1 +
                           df['trend_indicator'] * df['weight_trend'] +
                           df['avg_turnover'] * df['weight_liquidity'])

    return df['alpha_factor']
