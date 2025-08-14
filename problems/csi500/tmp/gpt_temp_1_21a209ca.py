import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Define lookback periods
    short_sma_period = 20
    long_sma_period = 50
    volatility_period = 30
    percent_change_period = 10
    turnover_period = 30
    reevaluation_frequency = 60  # Re-evaluate weights every 60 days

    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA_short'] = df['close'].rolling(window=short_sma_period).mean()
    df['SMA_long'] = df['close'].rolling(window=long_sma_period).mean()

    # Compute Volume-Adjusted Volatility
    df['high_low_diff'] = df['high'] - df['low']
    df['volume_weighted_volatility'] = df['high_low_diff'] * df['volume']
    df['volatility'] = df['volume_weighted_volatility'].rolling(window=volatility_period).mean()

    # Compute Price Momentum
    df['price_momentum'] = (df['close'] - df['SMA_short']) / df['close'].rolling(window=short_sma_period).mean()

    # Incorporate Additional Price Change Metrics
    df['percent_change'] = df['close'].pct_change(periods=percent_change_period)
    df['high_low_range'] = df['high'] - df['low']

    # Consider Market Trend Alignment
    df['trend_indicator'] = (df['SMA_short'] > df['SMA_long']).astype(int)

    # Incorporate Liquidity Measures
    df['daily_turnover'] = df['volume'] * df['close']
    df['turnover_avg'] = df['daily_turnover'].rolling(window=turnover_period).mean()

    # Define initial weights
    weights = {
        'price_momentum': 0.4,
        'volatility': -0.2,
        'percent_change': 0.2,
        'high_low_range': 0.1,
        'liquidity': 0.1
    }

    # Adjust weights based on market trend and liquidity
    def adjust_weights(row):
        if row['trend_indicator'] == 1:  # Bullish
            weights['price_momentum'] *= 1.2
            weights['volatility'] *= 0.8
        else:  # Bearish
            weights['price_momentum'] *= 0.8
            weights['volatility'] *= 1.2

        if row['turnover_avg'] > df['turnover_avg'].median():
            weights['price_momentum'] *= 1.1
            weights['percent_change'] *= 1.1
        else:
            weights['price_momentum'] *= 0.9
            weights['percent_change'] *= 0.9

        # Normalize weights
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight

        return pd.Series(weights)

    # Dynamically adjust weights
    df[['price_momentum', 'volatility', 'percent_change', 'high_low_range', 'liquidity']] = df.apply(adjust_weights, axis=1)

    # Final Alpha Factor
    df['alpha_factor'] = (
        df['price_momentum'] * df['price_momentum_weight'] +
        df['volatility'] * df['volatility_weight'] +
        df['percent_change'] * df['percent_change_weight'] +
        df['high_low_range'] * df['high_low_range_weight'] +
        df['turnover_avg'] * df['liquidity_weight']
    )

    # Re-evaluate weights at regular intervals
    df['alpha_factor'] = df['alpha_factor'].shift(reevaluation_frequency)

    return df['alpha_factor']
