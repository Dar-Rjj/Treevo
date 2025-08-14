import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday High-to-Low Range
    daily_range = (df['high'] - df['low']) / df['open']
    
    # Open to Close Momentum
    open_to_close_return = (df['close'] - df['open']) / df['open']
    smooth_open_to_close = open_to_close_return.ewm(span=5).mean()
    
    # Volume Adjusted Intraday Movement
    intraday_movement = df['close'] - df['open']
    avg_volume = df['volume'].rolling(window=10).mean()
    volume_adjusted_movement = intraday_movement / avg_volume
    
    # Price-Volume Trend Indicator
    price_change = df['close'] - df['close'].shift(1)
    price_volume_trend = price_change * df['volume']
    price_volume_trend_sum = price_volume_trend.rolling(window=30).sum()
    
    # Dynamic Weights based on recent performance
    def dynamic_weights(series, window=10):
        return series.rolling(window=window).corr(df['close']).fillna(0.5)
    
    weight_intraday_range = dynamic_weights(daily_range)
    weight_smooth_open_to_close = dynamic_weights(smooth_open_to_close)
    combined_momentum_volatility = (weight_intraday_range * daily_range + 
                                    weight_smooth_open_to_close * smooth_open_to_close)
    
    # Adaptive Exponential Smoothing for Combined Momentum and Volatility Factor
    alpha = 2 / (1 + df['close'].rolling(window=10).std().fillna(0))
    smoothed_combined_factor = combined_momentum_volatility.ewm(alpha=alpha).mean()
    
    # Volume-Sensitive Momentum Factor
    weight_price_volume_trend = dynamic_weights(price_volume_trend_sum)
    weight_volume_adjusted_movement = dynamic_weights(volume_adjusted_movement)
    volume_sensitive_momentum = (weight_price_volume_trend * price_volume_trend_sum + 
                                 weight_volume_adjusted_movement * volume_adjusted_movement)
    
    # Adaptive Exponential Smoothing for Volume-Sensitive Momentum Factor
    alpha = 2 / (1 + df['volume'].rolling(window=10).std().fillna(0))
    smoothed_volume_sensitive_momentum = volume_sensitive_momentum.ewm(alpha=alpha).mean()
    
    # Final Alpha Factor
    weight_combined_factor = dynamic_weights(smoothed_combined_factor)
    weight_volume_momentum = dynamic_weights(smoothed_volume_sensitive_momentum)
    final_alpha_factor = (weight_combined_factor * smoothed_combined_factor + 
                          weight_volume_momentum * smoothed_volume_sensitive_momentum)
    
    # Adaptive Exponential Smoothing for Final Alpha Factor
    alpha = 2 / (1 + df[['close', 'volume']].rolling(window=10).std().max(axis=1).fillna(0))
    final_alpha_factor = final_alpha_factor.ewm(alpha=alpha).mean()
    
    return final_alpha_factor
