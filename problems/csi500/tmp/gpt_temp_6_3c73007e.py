import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Acceleration Components
    # Short-term momentum (3-day)
    short_momentum = data['close'] / data['close'].shift(3) - 1
    
    # Medium-term momentum (5-day)
    medium_momentum = data['close'] / data['close'].shift(5) - 1
    
    # Price Acceleration Ratio
    price_acceleration = short_momentum / medium_momentum
    price_acceleration = price_acceleration.replace([np.inf, -np.inf], np.nan)
    
    # Calculate Volume Pattern Quality
    # Volume trend strength (10-day average)
    volume_avg = data['volume'].rolling(window=10, min_periods=1).mean()
    volume_trend = data['volume'] / volume_avg
    
    # Volume volatility context
    volume_std = data['volume'].rolling(window=10, min_periods=1).std()
    
    # Volume Spike Quality Score
    volume_quality = data['volume'] / (volume_avg + 2 * volume_std)
    volume_quality = volume_quality.replace([np.inf, -np.inf], np.nan)
    
    # Calculate Volatility-Adjusted Divergence
    # Daily returns for correlation calculation
    daily_returns = data['close'] / data['close'].shift(1) - 1
    
    # Price-Volume Correlation (10-day)
    price_volume_corr = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 9:
            window_returns = daily_returns.iloc[i-9:i+1]
            window_volume = data['volume'].iloc[i-9:i+1]
            if len(window_returns) == len(window_volume):
                corr = window_returns.corr(window_volume)
                price_volume_corr.iloc[i] = corr if not pd.isna(corr) else 0
    
    # Daily Range Volatility (10-day average)
    daily_range = data['high'] - data['low']
    range_volatility = daily_range.rolling(window=10, min_periods=1).mean()
    
    # Volatility-Adjusted Divergence Strength
    divergence_strength = (abs(price_acceleration) * abs(1 - price_volume_corr)) * range_volatility
    
    # Calculate Trend Persistence Confirmation
    # Directional Consistency (3-day)
    directional_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:
            returns_3d = daily_returns.iloc[i-2:i+1]
            if len(returns_3d) == 3:
                # Count consecutive same-sign returns
                signs = np.sign(returns_3d)
                if all(signs == signs.iloc[0]) and signs.iloc[0] != 0:
                    directional_consistency.iloc[i] = 3
                elif (signs.iloc[0] == signs.iloc[1] and signs.iloc[0] != 0) or (signs.iloc[1] == signs.iloc[2] and signs.iloc[1] != 0):
                    directional_consistency.iloc[i] = 2
                elif any(signs != 0):
                    directional_consistency.iloc[i] = 1
                else:
                    directional_consistency.iloc[i] = 0
    
    # Gap Persistence
    gap_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(1, len(data)):
        gap_magnitude = abs(data['open'].iloc[i] / data['close'].iloc[i-1] - 1)
        
        # Check gap direction consistency over 3 days
        if i >= 2:
            gaps = []
            for j in range(max(0, i-2), i+1):
                if j > 0:
                    gap = data['open'].iloc[j] / data['close'].iloc[j-1] - 1
                    gaps.append(gap)
            
            if len(gaps) >= 2:
                gap_signs = np.sign(gaps)
                if all(gap_signs == gap_signs[0]) and gap_signs[0] != 0:
                    streak_length = len(gaps)
                elif (gap_signs[0] == gap_signs[1]) and gap_signs[0] != 0:
                    streak_length = 2
                else:
                    streak_length = 1
            else:
                streak_length = 1
        else:
            streak_length = 1
        
        gap_persistence.iloc[i] = gap_magnitude * streak_length
    
    # Combined Persistence Signal
    persistence_signal = directional_consistency * gap_persistence
    
    # Generate Composite Alpha Factor
    # Apply directional alignment check
    direction_alignment = np.sign(short_momentum) * np.sign(volume_quality.fillna(0))
    
    # Final composite factor
    alpha_factor = (price_acceleration * 
                   volume_quality * 
                   divergence_strength * 
                   persistence_signal * 
                   direction_alignment)
    
    return alpha_factor
