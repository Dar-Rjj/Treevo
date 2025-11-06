import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Liquidity Momentum Component
    # 3-day liquidity momentum
    liquidity_3d = (data['volume'] * data['close']) / (data['volume'].shift(3) * data['close'].shift(3)) - 1
    
    # 8-day liquidity momentum
    liquidity_8d = (data['volume'] * data['close']) / (data['volume'].shift(8) * data['close'].shift(8)) - 1
    
    # Liquidity acceleration
    liquidity_acceleration = liquidity_3d - liquidity_8d
    
    # Price Range Asymmetry Component
    # Calculate daily range
    daily_range = data['high'] - data['low']
    
    # Identify up and down days
    daily_return = data['close'] / data['close'].shift(1) - 1
    up_days = daily_return > 0
    down_days = daily_return < 0
    
    # Calculate rolling averages for up and down day ranges
    up_day_range_avg = daily_range.rolling(window=5).apply(
        lambda x: x[up_days.loc[x.index].values].mean() if up_days.loc[x.index].sum() > 0 else np.nan, 
        raw=False
    )
    
    down_day_range_avg = daily_range.rolling(window=5).apply(
        lambda x: x[down_days.loc[x.index].values].mean() if down_days.loc[x.index].sum() > 0 else np.nan, 
        raw=False
    )
    
    # Up-day range ratio
    up_range_ratio = daily_range / up_day_range_avg
    
    # Down-day range intensity
    down_range_intensity = daily_range / down_day_range_avg
    
    # Range asymmetry index
    range_asymmetry = up_range_ratio - down_range_intensity
    
    # Market Regime Detection
    # Trade size concentration
    trade_size_concentration = data['amount'] / data['volume']
    
    # Liquidity efficiency
    liquidity_efficiency = data['amount'] / daily_range
    
    # Regime shift
    regime_shift = liquidity_efficiency / liquidity_efficiency.shift(5)
    
    # Factor Integration
    # Base factor
    base_factor = liquidity_acceleration * range_asymmetry
    
    # Regime weighting
    regime_weighted_factor = base_factor * regime_shift
    
    # Microstructure adjustment with trade size concentration filter
    # Normalize trade size concentration to create a filter between 0.5 and 1.5
    ts_normalized = (trade_size_concentration - trade_size_concentration.rolling(20).mean()) / trade_size_concentration.rolling(20).std()
    concentration_filter = 1 + 0.5 * np.tanh(ts_normalized / 3)
    
    # Final factor
    final_factor = regime_weighted_factor * concentration_filter
    
    return final_factor
