import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration
    short_term_momentum = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    medium_term_momentum = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    acceleration = (short_term_momentum - medium_term_momentum) / 7
    
    # Intraday Pressure
    high_low_range = data['high'] - data['low']
    # Avoid division by zero
    high_low_range = high_low_range.replace(0, np.nan)
    
    buying_pressure = ((data['close'] - data['low']) / high_low_range) * data['volume']
    selling_pressure = ((data['high'] - data['close']) / high_low_range) * data['volume']
    net_pressure = buying_pressure - selling_pressure
    
    # Volatility Breakout
    rolling_max_high = data['high'].rolling(window=5, min_periods=1).max()
    rolling_min_low = data['low'].rolling(window=5, min_periods=1).min()
    range_breakout = (data['close'] - rolling_max_high) * (data['close'] - rolling_min_low)
    
    # Avoid division by zero for volatility adjustment
    daily_volatility = (data['high'] - data['low']) / data['close']
    daily_volatility = daily_volatility.replace(0, np.nan)
    volatility_adjustment = range_breakout / daily_volatility
    
    # Flow Persistence
    # Direction Count - count same sign of daily returns in past 5 days
    daily_returns = data['close'].diff()
    sign_changes = daily_returns.rolling(window=5, min_periods=1).apply(
        lambda x: sum((x.iloc[1:] * x.iloc[:-1]) > 0) if len(x) > 1 else 0, raw=False
    )
    direction_count = 5 - sign_changes  # Count of same sign moves
    
    # Volume Trend
    volume_trend = np.sign(data['volume'] - data['volume'].shift(5))
    volume_trend = volume_trend.replace(0, 1)  # Treat zero difference as positive trend
    
    # Composite Alpha
    core_factor = acceleration * net_pressure * volatility_adjustment
    final_alpha = core_factor * direction_count * volume_trend
    
    # Fill NaN values with 0
    final_alpha = final_alpha.fillna(0)
    
    return final_alpha
