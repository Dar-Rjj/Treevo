import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts with adaptive window
    pos_sum = df['positive_amount_vol'].rolling(window=df['volume'].rolling(window=5).mean().astype(int)).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=df['volume'].rolling(window=5).mean().astype(int)).sum()

    # Factor: ratio of positive sum to absolute negative sum
    factor1 = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the 7-day and 21-day simple moving average (SMA)
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_21'] = df['close'].rolling(window=21).mean()

    # Mean reversion factor
    mean_reversion_factor = df['close'] - df['sma_7']

    # Volatility factor
    volatility = df['close'].pct_change().rolling(window=21).std()

    # Sentiment factor: ratio of close to SMA
    sentiment_factor = df['close'] / df['sma_21']

    # Combine factors using dynamic weights based on rolling correlation
    corr_mean_reversion = df['close'].rolling(window=21).corr(mean_reversion_factor)
    corr_volatility = df['close'].rolling(window=21).corr(volatility)
    corr_sentiment = df['close'].rolling(window=21).corr(sentiment_factor)

    total_corr = corr_mean_reversion + corr_volatility + corr_sentiment
    weight_mean_reversion = corr_mean_reversion / total_corr
    weight_volatility = corr_volatility / total_corr
    weight_sentiment = corr_sentiment / total_corr

    combined_factor = (weight_mean_reversion * mean_reversion_factor +
                       weight_volatility * volatility +
                       weight_sentiment * sentiment_factor +
                       factor1) / 4

    return combined_factor
