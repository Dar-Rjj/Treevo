import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    combined_factor = intraday_return * high_low_range
    
    # Smooth using Exponential Moving Average (EMA) with adaptive period
    volatility = df['close'].rolling(window=14).std()
    ema_period = (14 + 7 * (volatility / volatility.mean())).astype(int)
    smoothed_factor = df.rolling(window=ema_period, min_periods=1).mean()['close']
    
    # Apply Volume Weighting
    volume_weighted_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    previous_day_close = df['close'].shift(1)
    closing_gap = df['open'] - previous_day_close
    gap_adjusted_factor = volume_weighted_factor + closing_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Dynamic Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    atr = df[['high', 'low', 'close']].rolling(window=14).apply(lambda x: np.max(x) - np.min(x), raw=True)
    combined_volatility = (rolling_std + atr) / 2
    
    # Adjust Volatility Component with Volume
    volatility_adjusted = combined_volatility * df['volume']
    
    # Consider Liquidity
    avg_volume_50_days = df['volume'].rolling(window=50).mean()
    liquidity_indicator = df['volume'] / avg_volume_50_days
    final_factor = (gap_adjusted_factor + normalized_long_term_return + volatility_adjusted) * liquidity_indicator
    
    # Apply Non-Linear Transformation
    final_factor = np.log(1 + final_factor**2)
    
    return final_factor
