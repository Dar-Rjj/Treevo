import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, market_index_close):
    # Calculate Adaptive Short-Term Momentum
    short_term_momentum = df['close'].ewm(span=5).mean()
    
    # Calculate Adaptive Long-Term Momentum
    long_term_momentum = df['close'].ewm(span=20).mean()
    
    # Calculate Momentum Difference
    momentum_difference = short_term_momentum - long_term_momentum
    
    # Factor in Volume Trend
    volume_trend = df['volume'].ewm(span=10).mean()
    momentum_difference_with_volume = momentum_difference * (df['volume'] / volume_trend)
    
    # Incorporate Volatility
    true_range = df[['high', 'low']].apply(lambda x: np.max(x) - np.min(x), axis=1)
    daily_volatility = true_range.rolling(window=20).std()
    normalized_momentum_difference = momentum_difference_with_volume / daily_volatility
    
    # Incorporate Broad Market Data
    market_index_ema = market_index_close.ewm(span=20).mean()
    market_index_momentum_difference = market_index_close - market_index_ema
    final_alpha_factor = normalized_momentum_difference * market_index_momentum_difference
    
    return final_alpha_factor
