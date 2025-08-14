import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, k=10, m=10, n=5):
    # Calculate Adaptive Intraday Momentum
    df['DailyRange'] = df['High'] - df['Low']
    avg_daily_range = df['DailyRange'].rolling(window=k).mean()
    adaptive_lookback_period = (k * avg_daily_range / df['DailyRange']).astype(int)
    
    # Sum of Daily Ranges over Adaptive Lookback Period
    intraday_momentum = df['DailyRange'].rolling(window=adaptive_lookback_period, min_periods=1).sum()
    
    # Calculate Volume Surge
    avg_volume = df['Volume'].rolling(window=m).mean()
    volume_surge = (df['Volume'] > avg_volume).astype(int)
    
    # Integrate Relative Strength
    relative_strength = df['Close'] / df['Close'].shift(1).rolling(window=n).mean()
    
    # Incorporate Market Sentiment
    market_sentiment = np.where(df['Close'] > df['Open'], 1, -1)
    
    # Combine All Factors
    alpha_factor = intraday_momentum * volume_surge * relative_strength * market_sentiment
    
    return alpha_factor
