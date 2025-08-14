import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Intraday Midpoint
    intraday_midpoint = (df['open'] + df['close']) / 2
    
    # Determine Intraday Direction
    intraday_direction = (df['close'] > df['open']).astype(int) * 2 - 1  # 1 for bullish, -1 for bearish
    
    # Compute Intraday Momentum Reversal Score
    intraday_momentum_reversal_score = intraday_range * intraday_direction
    
    # Weight by Volume
    weighted_score = intraday_momentum_reversal_score * df['volume']
    
    # Incorporate Volume-Based Strengthening
    rolling_avg_volume = df['volume'].rolling(window=20).mean()
    volume_adjusted_score = weighted_score * (1.5 if df['volume'] > rolling_avg_volume else 1)
    
    # Incorporate Volatility Adjustment
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=20).std()
    normalized_score = volume_adjusted_score / volatility
    
    return normalized_score
