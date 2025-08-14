import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the difference between close and open prices, suggesting the direction of the day's movement
    daily_movement = df['close'] - df['open']
    
    # Compute the volatility as the average true range over a 10-day window for each date, indicating market turbulence
    tr = df[['high', 'low', 'close']].shift(1).apply(lambda x: max(x['high']-x['low'], abs(x['high']-x['close'].shift(1)), abs(x['low']-x['close'].shift(1))), axis=1)
    atr = tr.rolling(window=10).mean()
    
    # Calculate the ratio of daily movement to ATR, representing a normalized strength of the daily move against recent volatility
    movement_strength = daily_movement / (atr + 1e-7)
    
    return movement_strength
