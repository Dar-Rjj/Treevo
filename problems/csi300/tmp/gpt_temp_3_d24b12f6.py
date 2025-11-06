import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    price_range = high - low
    typical_price = (high + low + close) / 3
    
    bullish_intensity = ((close - low) / price_range) * volume
    bearish_intensity = ((high - close) / price_range) * volume
    
    sentiment_ratio = bullish_intensity.rolling(10).mean() / (bearish_intensity.rolling(10).mean() + 1e-8)
    
    sentiment_extreme = sentiment_ratio.rolling(20).apply(lambda x: (x[-1] - x.mean()) / x.std())
    
    price_momentum = typical_price / typical_price.rolling(5).mean() - 1
    
    divergence_signal = -price_momentum * np.sign(sentiment_extreme)
    
    volatility_filter = 1 / (price_range.rolling(10).std() + 1e-8)
    
    heuristics_matrix = divergence_signal * volatility_filter * np.abs(sentiment_extreme)
    
    return heuristics_matrix
