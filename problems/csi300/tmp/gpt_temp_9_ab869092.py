import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required components
    # Intraday Return
    intraday_return = data['close'] / data['open'] - 1
    
    # Daily Range
    daily_range = data['high'] - data['low']
    
    # 20-day Average Volume
    volume_20d_avg = data['volume'].rolling(window=20, min_periods=1).mean()
    
    # Volume Ratio
    volume_ratio = data['volume'] / volume_20d_avg
    
    # Breakout calculation (Close_t - Max(High_t-5 to t-1))
    high_5d_max = data['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    breakout = data['close'] - high_5d_max
    
    # Volume Efficiency
    volume_efficiency = data['amount'] / data['volume']
    
    # Opening Gap (Open_t / Close_t-1 - 1)
    prev_close = data['close'].shift(1)
    opening_gap = data['open'] / prev_close - 1
    
    # Intraday Momentum
    intraday_momentum = data['close'] / data['open'] - 1
    
    # Volume Spike detection (Volume_t > Percentile(Volume_t-20 to t-1, 95%))
    volume_20d_percentile = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: np.percentile(x[:-1], 95) if len(x) > 1 else np.nan
    )
    volume_spike = (data['volume'] > volume_20d_percentile).astype(float)
    
    # Price Reversal (Close_t / Max(High_t-5 to t-1) - 1)
    price_reversal = data['close'] / high_5d_max - 1
    
    # Calculate individual factors
    # Factor 1: Intraday Reversal with Volatility and Volume Adjustment
    factor1 = (intraday_return / daily_range.replace(0, np.nan)) * volume_ratio * (-1)
    
    # Factor 2: Range Breakout with Volume Efficiency
    factor2 = (breakout * volume_efficiency) / daily_range.replace(0, np.nan)
    
    # Factor 3: Gap Persistence with Momentum
    factor3 = opening_gap * intraday_momentum * volume_ratio
    
    # Factor 4: Volume-Spike Reversal with Range Adjustment
    factor4 = (price_reversal * volume_spike) / daily_range.replace(0, np.nan)
    
    # Combine factors (equal weighting)
    combined_factor = (factor1 + factor2 + factor3 + factor4) / 4
    
    # Handle NaN values by forward filling
    factor = combined_factor.fillna(method='ffill')
    
    return factor
