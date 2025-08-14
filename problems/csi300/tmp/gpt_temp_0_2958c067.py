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
    
    # Smooth using Exponential Moving Average (EMA)
    ema_period = 14
    smoothed_factor = combined_factor.ewm(span=ema_period, adjust=False).mean()
    
    # Apply Volume Weighting
    volume_weighted_smoothed_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    previous_day_close = df['close'].shift(1)
    closing_gap = df['open'] - previous_day_close
    volume_weighted_smoothed_factor_with_gap = volume_weighted_smoothed_factor + closing_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    volume_adjusted_volatility = rolling_std * df['volume']
    
    # Incorporate Market Sentiment (Assuming a market sentiment score is available in the DataFrame)
    # For this example, we assume a column 'sentiment_score' exists in the DataFrame
    market_sentiment_adjusted_factor = (volume_weighted_smoothed_factor_with_gap + 
                                        normalized_long_term_return + 
                                        volume_adjusted_volatility) * (1 + df['sentiment_score'])
    
    # Final Factor Calculation
    final_factor = (volume_weighted_smoothed_factor + 
                    closing_gap + 
                    normalized_long_term_return + 
                    volume_adjusted_volatility + 
                    market_sentiment_adjusted_factor)
    
    # Apply Non-Linear Transformation
    final_factor_transformed = np.log(1 + final_factor)
    
    return final_factor_transformed
