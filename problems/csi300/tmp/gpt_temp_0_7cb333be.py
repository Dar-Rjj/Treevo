import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the price change from the previous day to the current day
    price_change = df['close'] - df['close'].shift(1)
    
    # Calculate the average true range (ATR) as a measure of volatility over the last 14 days
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    tr = high_low.combine(high_close_prev, max).combine(low_close_prev, max)
    atr = tr.rolling(window=14).mean()
    
    # Calculate the weighted volume by the closing price
    weighted_volume = df['volume'] * df['close']
    
    # Calculate the ratio of today's weighted volume to the moving average of the weighted volume over a certain period
    moving_avg_weighted_volume = df['volume'].rolling(window=5).mean() * df['close'].rolling(window=5).mean()
    volume_ratio = weighted_volume / (moving_avg_weighted_volume + 1e-7)
    
    # Calculate the factor as the product of the price change, ATR, and the volume ratio
    factor_values = price_change * atr * volume_ratio
    
    return factor_values
