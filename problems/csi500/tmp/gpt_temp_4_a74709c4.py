import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum using Close Price
    short_sma = df['close'].rolling(window=10).mean()
    long_sma = df['close'].rolling(window=50).mean()
    momentum_signal = short_sma - long_sma
    
    # Analyze Volume Trends
    volume_change = df['volume'] - df['volume'].shift(1)
    volume_change_score = np.where(volume_change > 0, 1, -1)
    
    # Identify Volume Spikes
    avg_volume = df['volume'].rolling(window=50).mean()
    volume_spike_threshold = 2 * avg_volume
    volume_spike = np.where(df['volume'] > volume_spike_threshold, 1, 0)
    
    # Combine Price and Volume Indicators
    composite_factor = momentum_signal + volume_change_score
    composite_factor = (composite_factor - composite_factor.min()) / (composite_factor.max() - composite_factor.min())
    
    # Incorporate Price Patterns
    bullish_candle = np.where(df['close'] > df['open'], 1, 0)
    bearish_candle = np.where(df['close'] < df['open'], 1, 0)
    consecutive_bullish = bullish_candle.rolling(window=3).sum()
    consecutive_bearish = bearish_candle.rolling(window=3).sum()
    
    # Detect Reversal Patterns
    hammer_pattern = np.where((df['close'] - df['low']) > 2 * (df['high'] - df['close']), 1, 0)
    hanging_man_pattern = np.where((df['high'] - df['low']) > 2 * (df['high'] - df['close']), 1, 0)
    engulfing_pattern = np.where((df['close'] > df['open'].shift(1)) & (df['close'].shift(1) < df['open']), 1, 0)
    
    # Evaluate Market Sentiment
    daily_range = df['high'] - df['low']
    range_expansion = np.where(daily_range > daily_range.shift(1), 1, -1)
    
    # Combine Range and Price Movement
    combined_sentiment_factor = range_expansion + momentum_signal
    combined_sentiment_factor = (combined_sentiment_factor - combined_sentiment_factor.min()) / (combined_sentiment_factor.max() - combined_sentiment_factor.min())
    
    # Final Alpha Factor
    alpha_factor = composite_factor + combined_sentiment_factor + (consecutive_bullish - consecutive_bearish) + hammer_pattern + hanging_man_pattern + engulfing_pattern
    alpha_factor = (alpha_factor - alpha_factor.min()) / (alpha_factor.max() - alpha_factor.min())
    
    return alpha_factor
