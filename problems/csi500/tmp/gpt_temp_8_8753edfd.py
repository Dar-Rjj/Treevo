import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = (df['high'] - df['low']) / df['open']
    
    # Adjust for Volume
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_adjustment = (df['volume'] - volume_ma_10) * intraday_return
    
    # Incorporate Price Volatility
    close_std_5 = df['close'].rolling(window=5).std()
    volatility_factor = 1.2 if close_std_5 > close_std_5.mean() else 0.8
    adjusted_intraday_return = volume_adjustment * volatility_factor
    
    # Calculate Momentum Score
    lookback_days = 20
    close_change = df['close'] - df['close'].shift(lookback_days)
    momentum_score = close_change.rolling(window=lookback_days).sum()
    
    # Adjust for Volume Volatility
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_deviation = df['volume'] - volume_ma_20
    
    # Combine Intraday Return and Momentum Score
    combined_score = adjusted_intraday_return + momentum_score
    
    # Apply Adjustment to Combined Score
    final_score = combined_score * volume_deviation
    
    return final_score
