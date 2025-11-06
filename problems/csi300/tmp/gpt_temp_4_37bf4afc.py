import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Volatility-Adjusted Intraday Reversal
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, np.nan)  # Avoid division by zero
    
    reversal_high = (data['high'] - data['close']) / high_low_range * data['volume']
    reversal_low = (data['close'] - data['low']) / high_low_range * data['volume']
    vol_adjusted_reversal = reversal_high - reversal_low
    
    # Dynamic Breakout with Efficiency
    roll_max_high = data['high'].rolling(window=20, min_periods=1).max()
    roll_min_low = data['low'].rolling(window=20, min_periods=1).min()
    
    close_open_abs = abs(data['close'] - data['open'])
    price_efficiency = close_open_abs / high_low_range.replace(0, np.nan)
    
    breakout_high = (data['close'] - roll_max_high) * price_efficiency
    breakout_low = (data['close'] - roll_min_low) * price_efficiency
    dynamic_breakout = breakout_high - breakout_low
    
    # Asymmetric Volatility Momentum
    returns = data['close'].pct_change()
    
    # Calculate upside and downside volatility over 20 days
    upside_vol = returns.rolling(window=20).apply(
        lambda x: x[x > 0].std() if len(x[x > 0]) > 1 else 0, raw=False
    )
    downside_vol = returns.rolling(window=20).apply(
        lambda x: x[x < 0].std() if len(x[x < 0]) > 1 else 0, raw=False
    )
    
    vol_ratio = upside_vol / downside_vol.replace(0, np.nan)
    asymmetric_vol_momentum = vol_ratio * returns
    
    # Volume-Weighted Price Pattern
    price_acceleration = returns.diff()
    volume_acceleration = data['volume'] / data['volume'].shift(1)
    
    acceleration_factor = (price_acceleration * volume_acceleration).rolling(window=5).sum()
    
    # Gap Momentum with Position Analysis
    prev_close = data['close'].shift(1)
    absolute_gap = (data['open'] / prev_close - 1).abs()
    intraday_momentum = data['close'] / data['open'] - 1
    position_ratio = (data['close'] - data['low']) / high_low_range.replace(0, np.nan)
    
    gap_momentum = absolute_gap * intraday_momentum * position_ratio
    
    # Support Resistance with Volume Confirmation
    lowest_low_20 = data['low'].rolling(window=20, min_periods=1).min()
    distance_to_support = data['close'] / lowest_low_20 - 1
    
    # Calculate support volume (volume when price is near 20-day low)
    near_support_threshold = 0.02  # 2% threshold
    near_support = (data['low'] <= lowest_low_20 * (1 + near_support_threshold))
    support_volume = data['volume'].where(near_support, 0).rolling(window=5).mean()
    
    support_factor = distance_to_support * support_volume
    
    # Combine all factors with equal weights
    factor = (
        vol_adjusted_reversal.rank(pct=True) +
        dynamic_breakout.rank(pct=True) +
        asymmetric_vol_momentum.rank(pct=True) +
        acceleration_factor.rank(pct=True) +
        gap_momentum.rank(pct=True) +
        support_factor.rank(pct=True)
    ) / 6
    
    return factor
