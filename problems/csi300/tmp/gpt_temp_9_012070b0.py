import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['next_open'] = df['open'].shift(-1)
    df['simple_returns'] = (df['next_open'] - df['close']) / df['close']
    df['volume_weighted_returns'] = df['simple_returns'] * df['volume']
    
    # Identify Volume Surge Days
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['rolling_vol_mean'] = df['volume'].rolling(window=5).mean()
    df['volume_surge'] = df['volume'] > df['rolling_vol_mean']
    
    # Calculate Volatility
    df['daily_returns'] = df['close'].pct_change()
    N = 20  # Lookback period for volatility
    df['volatility'] = df['daily_returns'].rolling(window=N).std()
    
    # Adjust Volume-Weighted Returns by Volatility
    df['adjusted_returns'] = df['volume_weighted_returns'] / df['volatility']
    
    # Combine Adjusted Returns with Volume Surge Indicator
    surge_factor = 1.5
    df.loc[df['volume_surge'], 'adjusted_returns'] *= surge_factor
    
    # Incorporate Momentum and Moving Averages
    short_ema_span = 5
    long_ema_span = 20
    df['short_ema'] = df['close'].ewm(span=short_ema_span, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=long_ema_span, adjust=False).mean()
    df['momentum'] = df['short_ema'] - df['long_ema']
    
    # Integrate with Dynamic Weighting Based on Trend Strength
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    trend_threshold = 25
    df['trend_strength_weight'] = np.where(df['adx'] > trend_threshold, 1.5, 1.0)
    df['final_alpha'] = df['adjusted_returns'] + df['trend_strength_weight'] * df['momentum']
    
    return df['final_alpha'].dropna()
