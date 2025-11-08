import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on asymmetric price dynamics and volume structure
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate values
    data['PrevClose'] = data['close'].shift(1)
    data['PrevVolume'] = data['volume'].shift(1)
    
    for i in range(len(data)):
        if i < 1:  # Skip first row due to lagged values
            result.iloc[i] = 0
            continue
            
        # Current day values
        open_price = data['open'].iloc[i]
        high = data['high'].iloc[i]
        low = data['low'].iloc[i]
        close = data['close'].iloc[i]
        volume = data['volume'].iloc[i]
        prev_close = data['PrevClose'].iloc[i]
        prev_volume = data['PrevVolume'].iloc[i]
        
        # 1. Intraday Momentum Asymmetry
        if (high - low) > 0:
            bullish_momentum = (close - open_price) / (high - low)
            bearish_momentum = (open_price - close) / (high - low)
            momentum_asymmetry = bullish_momentum - bearish_momentum
        else:
            momentum_asymmetry = 0
        
        # 2. Volume Concentration at Price Extremes
        # Simplified approximation using price position relative to mid-point
        mid_price = (open_price + close) / 2
        if close > mid_price:
            high_side_ratio = 1.5  # Approximation for volume concentration
            low_side_ratio = 0.5
        elif close < mid_price:
            high_side_ratio = 0.5
            low_side_ratio = 1.5
        else:
            high_side_ratio = 1.0
            low_side_ratio = 1.0
        
        if low_side_ratio > 0:
            volume_concentration = high_side_ratio / low_side_ratio
        else:
            volume_concentration = 1.0
        
        # 3. Opening Gap Persistence Pattern
        if prev_close > 0 and abs(open_price - prev_close) > 0:
            gap_direction = np.sign(open_price - prev_close) * np.sign(close - open_price)
            gap_magnitude = abs(close - open_price) / abs(open_price - prev_close)
            gap_persistence = gap_direction * gap_magnitude
        else:
            gap_persistence = 0
        
        # 4. Price Rejection Strength at Boundaries
        if (high - low) > 0:
            upper_rejection = (high - max(open_price, close)) / (high - low)
            lower_rejection = (min(open_price, close) - low) / (high - low)
            rejection_strength = upper_rejection - lower_rejection
        else:
            rejection_strength = 0
        
        # 5. Volume-Weighted Price Acceleration
        if prev_close > 0 and prev_volume > 0:
            price_momentum = (close - prev_close) / prev_close
            volume_acceleration = volume / prev_volume
            volume_weighted_acceleration = price_momentum * volume_acceleration
        else:
            volume_weighted_acceleration = 0
        
        # 6. Trading Range Efficiency Score
        if (high - low) > 0:
            price_efficiency = (close - open_price) / (high - low)
            volume_efficiency = volume / (high - low)
            range_efficiency = price_efficiency * volume_efficiency
        else:
            range_efficiency = 0
        
        # Combine all signals with equal weights
        combined_signal = (
            momentum_asymmetry +
            volume_concentration +
            gap_persistence +
            rejection_strength +
            volume_weighted_acceleration +
            range_efficiency
        ) / 6.0
        
        result.iloc[i] = combined_signal
    
    return result
