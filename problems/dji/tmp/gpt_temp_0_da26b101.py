import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Price Velocity
    price_velocity = df['close'].diff()
    
    # Calculate Short-Term EMA
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    
    # Calculate Long-Term EMA
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    
    # Calculate EMA Difference
    ema_diff = short_ema - long_ema
    
    # Calculate Short-Term Momentum
    short_momentum = df['close'].shift(1) - df['close'].shift(7)
    
    # Calculate Long-Term Momentum
    long_momentum = df['close'].shift(1) - df['close'].shift(30)
    
    # Calculate Daily Volume Change
    volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Adjust Momentum by Volume Shock
    adjusted_momentum = (short_momentum - long_momentum) * volume_change
    
    # Calculate Intraday High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Open-to-Close Momentum
    open_close_momentum = df['open'] - df['close']
    
    # Integrate Intraday and Open-to-Close Momentum
    integrated_intraday_momentum = (high_low_range + open_close_momentum) / 2
    
    # Synthesize Final Alpha Factor
    final_factor = (price_velocity + adjusted_momentum + ema_diff + integrated_intraday_momentum)
    
    # Apply Volume Filter
    high_volume_threshold = df['volume'].rolling(window=5).mean().quantile(0.8)
    high_volume_days = df['volume'] > high_volume_threshold
    final_factor_filtered = final_factor.where(high_volume_days, 0)
    
    # Incorporate Price and Volume Correlation
    price_volume_corr = df[['close', 'volume']].rolling(window=5).corr().iloc[::2, -1].reset_index(level=0, drop=True)
    final_factor_combined = 0.4 * final_factor_filtered + 0.6 * price_volume_corr
    
    return final_factor_combined
