import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=20, M=50):
    # Calculate Log Returns
    log_returns = np.log(df['close']).diff()
    
    # Compute Momentum
    momentum = log_returns.rolling(window=N).sum()
    upper_threshold = momentum.quantile(0.75)
    lower_threshold = momentum.quantile(0.25)
    momentum_clipped = momentum.clip(lower=lower_threshold, upper=upper_threshold)
    
    # Adjust for Volume
    volume_mean = df['volume'].rolling(window=M).mean()
    volume_adjusted_momentum = momentum_clipped * (df['volume'] / volume_mean)
    
    # Determine Absolute Price Changes
    abs_price_changes = df['close'].diff().abs()
    
    # Calculate Advanced Volatility Measures
    std_abs_price_changes = abs_price_changes.rolling(window=M).std()
    ema_abs_price_changes = abs_price_changes.ewm(span=M, adjust=False).mean()
    iqr_abs_price_changes = abs_price_changes.rolling(window=M).quantile(0.75) - abs_price_changes.rolling(window=M).quantile(0.25)
    
    # Integrate Market Sentiment
    # Assuming `positive_news_count` and `negative_news_count` are available in the DataFrame
    sentiment_index = (df['positive_news_count'] - df['negative_news_count']) / (df['positive_news_count'] + df['negative_news_count'])
    sentiment_normalized = (sentiment_index - sentiment_index.min()) / (sentiment_index.max() - sentiment_index.min())
    combined_momentum_sentiment = 0.6 * momentum_clipped + 0.4 * sentiment_normalized
    
    # Refined Volatility Measures
    high_low_spread = df['high'] - df['low']
    std_high_low_spread = high_low_spread.rolling(window=M).std()
    open_close_spread = df['open'] - df['close']
    std_open_close_spread = open_close_spread.rolling(window=M).std()
    
    # Integrate Economic Indicators
    # Assuming `interest_rate` and `gdp_growth_rate` are available in the DataFrame
    economic_indicator = df['interest_rate'] + df['gdp_growth_rate']
    
    # Final Factor Calculation
    final_factor = (
        0.4 * volume_adjusted_momentum +
        0.3 * std_abs_price_changes +
        0.1 * ema_abs_price_changes +
        0.1 * iqr_abs_price_changes +
        0.1 * combined_momentum_sentiment +
        0.05 * std_high_low_spread +
        0.05 * std_open_close_spread
    )
    
    return final_factor
