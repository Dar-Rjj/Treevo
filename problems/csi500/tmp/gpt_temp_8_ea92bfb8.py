import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, market_index):
    # Calculate High-Low Price Ratio
    df['High_Low_Ratio'] = df['high'] / df['low']
    
    # Calculate Weighted High-Low Momentum
    weights = [5, 4, 3, 2, 1]
    high_low_momentum = (df['High_Low_Ratio'].rolling(window=5).apply(lambda x: (x * weights).sum(), raw=True)).shift(1)
    
    # Dynamic Volume Correction Factor
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    median_volume_20 = df['volume'].rolling(window=20).median()
    volume_correction = avg_volume_20 / median_volume_20
    
    # Calculate Price Volatility
    log_returns = np.log(df['close'] / df['close'].shift(1))
    price_volatility = log_returns.rolling(window=20).std()
    
    # Smooth Volume Changes
    smoothed_volume = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Market Sentiment Indicator
    daily_range = df['high'] - df['low']
    gap = df['open'] - df['close'].shift(1)
    directional_movement = df['close'] - df['open']
    sentiment_score = (daily_range + gap + directional_movement) / 3
    
    # Broad Market Indicator
    market_log_returns = np.log(market_index['close'] / market_index['close'].shift(1))
    market_return = market_index['close'].pct_change()
    market_volatility = market_log_returns.rolling(window=20).std()
    
    # Combine all factors
    alpha_factor = (high_low_momentum * volume_correction / price_volatility * 
                    smoothed_volume * sentiment_score * market_return / market_volatility)
    
    return alpha_factor
