import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Delta
    high_low_delta = df['high'] - df['low']
    
    # Calculate Close-Open Delta
    close_open_delta = df['close'] - df['open']
    
    # Combine Deltas into Enhanced Momentum Score
    momentum_score = high_low_delta + close_open_delta
    
    # Weighted by Previous Day's Momentum Score with Exponential Decay
    previous_momentum_score = momentum_score.shift(1)
    decay_factor = 0.95
    momentum_score = (momentum_score + previous_momentum_score * decay_factor) / (1 + decay_factor)
    
    # Calculate Volume Change
    volume_change = df['volume'] - df['volume'].shift(1)
    
    # Apply Volume Weighting
    volume_weighted_momentum = momentum_score * volume_change
    
    # Threshold Filter to Select Significant Signals
    min_volume_increase_threshold = 1.5
    min_momentum_score_threshold = 0.1
    significant_signals = (volume_change > min_volume_increase_threshold) & (momentum_score > min_momentum_score_threshold)
    volume_weighted_momentum = volume_weighted_momentum * significant_signals
    
    # Calculate Volume Adjustment Factor
    long_term_avg_volume = df['volume'].rolling(window=20).mean()
    short_term_avg_volume = df['volume'].rolling(window=5).mean()
    volume_adjustment_factor = short_term_avg_volume / long_term_avg_volume
    
    # Integrate Price and Volume Changes with Efficiency
    price_change = df['close'] - df['close'].shift(1)
    price_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    integrated_indicator = (price_change * volume_change * price_efficiency) * volume_adjustment_factor
    
    # Cumulative Enhanced Momentum Indicator
    cumulative_enhanced_momentum = integrated_indicator.rolling(window=20).sum()
    
    # Smooth the Indicator
    ema_cumulative_enhanced_momentum = cumulative_enhanced_momentum.ewm(span=20, adjust=False).mean()
    
    # Enhance with Trend Analysis
    long_term_trend = df['close'].ewm(span=200, adjust=False).mean()
    short_term_trend = df['close'].ewm(span=50, adjust=False).mean()
    
    trend_factor = np.where(short_term_trend > long_term_trend, ema_cumulative_enhanced_momentum * 1.2, ema_cumulative_enhanced_momentum * 0.8)
    
    # Smooth the Adjusted Indicator
    final_factor = trend_factor.ewm(span=20, adjust=False).mean()
    
    # Incorporate Volume Volatility and Price Oscillation
    long_term_volatility_volume = df['volume'].rolling(window=20).std()
    short_term_volatility_volume = df['volume'].rolling(window=5).std()
    volatility_volume_ratio = short_term_volatility_volume / long_term_volatility_volume
    
    long_term_volatility_price = df['close'].rolling(window=20).std()
    short_term_volatility_price = df['close'].rolling(window=5).std()
    volatility_price_ratio = short_term_volatility_price / long_term_volatility_price
    
    final_factor = final_factor * volatility_volume_ratio * volatility_price_ratio
    
    return final_factor
