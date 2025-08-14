import pandas as pd
import pandas as pd

def heuristics_v2(df, short_sma_period=20, long_sma_period=200, vol_lookback=20, trend_lookback=20, liquidity_lookback=20):
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA_Short'] = df['close'].rolling(window=short_sma_period).mean()
    df['SMA_Long'] = df['close'].rolling(window=long_sma_period).mean()

    # Compute Volume-Adjusted Volatility
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Volume_Adjusted_Volatility'] = (df['High_Low_Diff'] * df['volume']).rolling(window=vol_lookback).mean()

    # Compute Price Momentum
    df['Price_Momentum'] = (df['close'] - df['SMA_Short']) / df['close'].rolling(window=short_sma_period).mean()

    # Incorporate Additional Price Change Metrics
    df['Pct_Change_Close'] = df['close'].pct_change(trend_lookback)
    df['High_Low_Range'] = df['high'] - df['low']

    # Consider Market Trend Alignment
    df['Trend_Indicator'] = (df['SMA_Short'] > df['SMA_Long']).astype(int)

    # Incorporate Dynamic Liquidity Measures
    df['Daily_Turnover'] = df['volume'] * df['close']
    df['Rolling_Avg_Turnover'] = df['Daily_Turnover'].rolling(window=liquidity_lookback).mean()
    df['Liquidity_Factor'] = df['Rolling_Avg_Turnover'] / df['Close_Price'].rolling(window=liquidity_lookback).mean()

    # Final Alpha Factor
    price_momentum_weight = 0.3
    volatility_weight = 0.2
    trend_weight = 0.1
    liquidity_weight = 0.2
    additional_metrics_weight = 0.2

    df['Alpha_Factor'] = (
        price_momentum_weight * df['Price_Momentum'] +
        volatility_weight * df['Volume_Adjusted_Volatility'] +
        trend_weight * df['Trend_Indicator'] +
        liquidity_weight * df['Liquidity_Factor'] +
        additional_metrics_weight * df['Pct_Change_Close']
    )

    # Adjust Weights Dynamically Based on Market Trend
    df['Alpha_Factor'] = df.apply(
        lambda row: row['Alpha_Factor'] * 1.1 if row['Trend_Indicator'] == 1 else row['Alpha_Factor'] * 0.9,
        axis=1
    )

    return df['Alpha_Factor'].dropna()

# Example usage:
# alpha_factor = heuristics_v2(df)
