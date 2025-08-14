import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-to-Low Momentum
    df['High_to_Low_Momentum'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Short-Term Trend
    short_term_trend_5 = df['High_to_Low_Momentum'].rolling(window=5).mean()
    short_term_trend_10 = df['High_to_Low_Momentum'].rolling(window=10).mean()
    
    # Calculate Long-Term Trend
    long_term_trend_20 = df['High_to_Low_Momentum'].rolling(window=20).mean()
    long_term_trend_50 = df['High_to_Low_Momentum'].rolling(window=50).mean()
    
    # Compute Short-Term and Long-Term Trend Difference
    short_term_diff = short_term_trend_5 - short_term_trend_10
    long_term_diff = long_term_trend_20 - long_term_trend_50
    
    # Weight by Volume
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_weight = df['volume'] / volume_ma_10
    
    # Incorporate Volatility
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=10).std()
    
    # Integrate Sentiment
    # Assuming sentiment scores are available in the DataFrame as 'sentiment_score'
    normalized_sentiment = (df['sentiment_score'] - df['sentiment_score'].min()) / (df['sentiment_score'].max() - df['sentiment_score'].min())
    
    # Combine Trends and Weights
    short_term_combined = short_term_diff * volume_weight
    long_term_combined = long_term_diff * volume_weight
    
    # Final Alpha Factor
    alpha_factor = short_term_combined + long_term_combined + 0.5 * volatility + 0.3 * normalized_sentiment
    
    # Multi-Frequency Momentum
    high_freq_momentum = df['High_to_Low_Momentum'].rolling(window=1).mean()
    medium_freq_momentum = df['High_to_Low_Momentum'].rolling(window=5).mean()
    low_freq_momentum = df['High_to_Low_Momentum'].rolling(window=20).mean()
    multi_freq_momentum = (high_freq_momentum + medium_freq_momentum + low_freq_momentum) / 3
    
    # Integrate with Final Alpha Factor
    alpha_factor += 0.2 * multi_freq_momentum
    
    return alpha_factor
