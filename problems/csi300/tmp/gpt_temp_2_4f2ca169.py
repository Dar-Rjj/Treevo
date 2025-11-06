import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence Momentum factor combining multiple timeframes,
    volume confirmation, and range efficiency metrics.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Multi-timeframe Momentum
    mom_5d = close / close.shift(5) - 1
    mom_10d = close / close.shift(10) - 1
    mom_ratio = (close / close.shift(5)) / (close / close.shift(10))
    
    # Volume Confirmation
    vol_trend = volume / volume.shift(5)
    vol_accel = (volume / volume.shift(5)) / (volume.shift(5) / volume.shift(10))
    
    # Volume persistence (count of days with volume > previous day's volume)
    vol_persistence = pd.Series(index=volume.index, dtype=float)
    for i in range(5, len(volume)):
        window = volume.iloc[i-5:i+1]
        vol_persistence.iloc[i] = (window > window.shift(1)).sum()
    
    # Range Efficiency Metrics
    daily_range = high - low
    gap_adjusted = np.maximum(high, close.shift(1)) - np.minimum(low, close.shift(1))
    normalized_range = (high - low) / close.shift(1)
    
    price_efficiency = np.abs(close - close.shift(1)) / (high - low).replace(0, np.nan)
    
    # 3-day cumulative efficiency
    cum_price_move = (np.abs(close - close.shift(1)) + 
                     np.abs(close.shift(1) - close.shift(2)) + 
                     np.abs(close.shift(2) - close.shift(3)))
    cum_range = (high - low + high.shift(1) - low.shift(1) + 
                high.shift(2) - low.shift(2))
    efficiency_3d = cum_price_move / cum_range.replace(0, np.nan)
    
    # Efficiency trend (current vs 5-day average)
    efficiency_avg = price_efficiency.rolling(window=5).mean().shift(1)
    efficiency_trend = price_efficiency / efficiency_avg
    
    # Divergence Signal Components
    volume_growth = volume / volume.shift(5)
    signal_strength = mom_5d * (1 / volume_growth.replace(0, np.nan))
    
    # Efficient Momentum Integration
    efficient_momentum = mom_5d * price_efficiency
    
    # Range Breakout/Breakdown signals
    range_breakout = ((close > high.shift(1)) & (price_efficiency > price_efficiency.rolling(10).mean())).astype(int)
    range_breakdown = ((close < low.shift(1)) & (price_efficiency > price_efficiency.rolling(10).mean())).astype(int)
    range_signal = range_breakout - range_breakdown
    
    # Combined Factor Construction
    factor = (
        mom_5d * 0.3 +                    # Short-term momentum
        mom_ratio * 0.2 +                 # Momentum ratio
        signal_strength * 0.25 +          # Divergence strength
        efficient_momentum * 0.15 +       # Efficient momentum
        range_signal * 0.1                # Range signals
    )
    
    # Volume persistence adjustment
    vol_weight = vol_persistence / 5.0  # Normalize to 0-1 range
    factor = factor * (1 + 0.1 * vol_weight)  # Boost factor with volume persistence
    
    # Normalize the final factor
    factor = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
    
    return factor
