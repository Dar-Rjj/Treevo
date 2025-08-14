import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    intraday_momentum = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']
    
    # Weighted Sum of Intraday and Close-to-Open Returns
    weighted_intraday = intraday_momentum * abs(intraday_momentum)
    weighted_close_to_open = close_to_open_return * abs(close_to_open_return)
    preliminary_factor = weighted_intraday + weighted_close_to_open
    
    # Calculate True Range (TR)
    df['previous_close'] = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['previous_close']).abs()
    tr3 = (df['low'] - df['previous_close']).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Positive Directional Movement (+DM) and Negative Directional Movement (-DM)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    positive_dm = (df['high'] - df['prev_high']).where((df['high'] > df['prev_high']) & ((df['high'] - df['prev_high']) > (df['prev_low'] - df['low'])), 0)
    negative_dm = (df['prev_low'] - df['low']).where((df['prev_low'] > df['low']) & ((df['prev_low'] - df['low']) > (df['high'] - df['prev_high'])), 0)
    
    # Smooth +DM and -DM, then divide by TR to get +DI and -DI
    smooth_period = 14
    positive_di = positive_dm.rolling(window=smooth_period).sum() / true_range.rolling(window=smooth_period).sum()
    negative_di = negative_dm.rolling(window=smooth_period).sum() / true_range.rolling(window=smooth_period).sum()
    
    # Calculate ADMI
    admi = (positive_di - negative_di) / (positive_di + negative_di)
    
    # Multiply Preliminary Factor by ADMI
    intermediate_alpha_factor = preliminary_factor * admi
    
    # Analyze Close Price Trends
    close_direction = np.where(df['close'] > df['previous_close'], 1, -1)
    
    # Examine the Relationship Between Close Price and Volume
    trailing_volume_window = 5
    avg_volume = df['volume'].rolling(window=trailing_volume_window).mean()
    price_vol_ratio = df['close'] / avg_volume
    
    # Look at Intra-Day Price Action
    intra_day_range = df['high'] - df['low']
    above_prev_close = (df['high'] - df['previous_close']) / df['previous_close']
    
    # Consider the Open-to-Close Behavior
    open_close_diff = df['close'] - df['open']
    day_move_strength = open_close_diff * (df['volume'] / avg_volume)
    
    # Evaluate the High and Low Prices
    high_streak = (df['high'] > df['high'].shift(1)).groupby(df.index.normalize()).cumsum().fillna(0)
    low_streak = (df['low'] < df['low'].shift(1)).groupby(df.index.normalize()).cumsum().fillna(0)
    
    # Assess the Volatility
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=trailing_volume_window).std()
    
    # Incorporate Volume-Sensitive Indicators
    volume_ratio = df['volume'] / avg_volume
    adjusted_alpha_factor = intermediate_alpha_factor * volume_ratio / (volatility + 1e-6)
    
    # Combine Intermediate Alpha Factor with New Components
    final_alpha_factor = adjusted_alpha_factor * close_direction * price_vol_ratio * (intra_day_range / df['previous_close']) * day_move_strength * (high_streak - low_streak)
    
    return final_alpha_factor
