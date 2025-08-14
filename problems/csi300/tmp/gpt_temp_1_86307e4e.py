import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    combined_factor = intraday_return * high_low_range
    smoothed_factor = combined_factor.ewm(span=14).mean()
    
    # Apply Volume Weighting
    volume_weighted_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    closing_gap = df['open'].shift(-1) - df['close']
    volume_weighted_smoothed_factor = volume_weighted_factor + closing_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Enhanced Dynamic Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    atr = (df['high'] - df['low']).rolling(window=14).mean()
    combined_volatility = (rolling_std + atr) / 2
    
    # Adjust Volatility Component with Volume
    adjusted_volatility = combined_volatility * df['volume']
    
    # Incorporate Trading Volume Trends
    rolling_avg_volume = df['volume'].rolling(window=20).mean()
    volume_trend = df['volume'] - rolling_avg_volume
    
    # Integrate Sentiment Analysis
    sentiment_score = df['sentiment']  # Assuming 'sentiment' column is available
    normalized_sentiment = (sentiment_score - sentiment_score.min()) / (sentiment_score.max() - sentiment_score.min())
    
    # Final Factor Calculation
    final_factor = (volume_weighted_smoothed_factor + 
                    closing_gap + 
                    normalized_long_term_return + 
                    adjusted_volatility + 
                    volume_trend + 
                    normalized_sentiment)
    
    # Apply Non-Linear Transformation
    final_factor = np.log(1 + final_factor)
    
    return final_factor
