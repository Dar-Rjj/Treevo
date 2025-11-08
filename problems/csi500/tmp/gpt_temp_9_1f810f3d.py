import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Absolute Momentum with Volume-Regime Alignment alpha factor
    
    Parameters:
    data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
    Series: Alpha factor values indexed by date
    """
    
    # Extract price and volume data
    open_prices = data['open']
    high_prices = data['high']
    low_prices = data['low']
    close_prices = data['close']
    volume = data['volume']
    
    # Initialize result series
    alpha_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate components for each day
    for i in range(10, len(data)):
        current_date = data.index[i]
        
        # Raw Price Components
        intraday_momentum = close_prices.iloc[i] - open_prices.iloc[i]
        daily_range = high_prices.iloc[i] - low_prices.iloc[i]
        
        # Absolute Momentum Calculation
        # Very Short-Term (1-2 days)
        momentum_1d = intraday_momentum
        momentum_2d = (close_prices.iloc[i] - close_prices.iloc[i-1]) + (close_prices.iloc[i-1] - close_prices.iloc[i-2])
        
        # Short-Term (5 days)
        momentum_5d = sum(close_prices.iloc[j] - open_prices.iloc[j] for j in range(i-4, i+1))
        
        # Medium-Term (10 days)
        momentum_10d = sum(close_prices.iloc[j] - open_prices.iloc[j] for j in range(i-9, i+1))
        
        # Volume Analysis
        volume_change = volume.iloc[i] - volume.iloc[i-1]
        volume_direction = np.sign(volume_change)
        momentum_direction = np.sign(intraday_momentum)
        
        # Regime Detection
        # Volatility Regime
        range_5d = np.mean([high_prices.iloc[j] - low_prices.iloc[j] for j in range(i-4, i+1)])
        range_10d = np.mean([high_prices.iloc[j] - low_prices.iloc[j] for j in range(i-9, i+1)])
        volatility_ratio = range_5d / range_10d if range_10d != 0 else 1.0
        
        # Volume Regime
        volume_5d_avg = np.mean([volume.iloc[j] for j in range(i-4, i+1)])
        volume_ratio = volume.iloc[i] / volume_5d_avg if volume_5d_avg != 0 else 1.0
        
        # Factor Construction
        # Base Momentum
        weighted_momentum = (5 * momentum_1d + 3 * momentum_5d + 2 * momentum_10d) / 10
        
        # Volume Alignment
        direction_alignment = np.sign(weighted_momentum) * volume_direction
        alignment_strength = direction_alignment * abs(volume_change)
        
        # Regime Scaling
        # Volatility Scaling
        if volatility_ratio > 1.1:
            volatility_scaling = 0.8
        elif volatility_ratio >= 0.9:
            volatility_scaling = 1.0
        else:
            volatility_scaling = 1.2
        
        # Volume Scaling
        if volume_ratio > 1.1:
            volume_scaling = 1.1
        elif volume_ratio >= 0.9:
            volume_scaling = 1.0
        else:
            volume_scaling = 0.9
        
        # Final Alpha
        alpha_value = weighted_momentum * alignment_strength * volatility_scaling * volume_scaling
        
        alpha_values.loc[current_date] = alpha_value
    
    return alpha_values
