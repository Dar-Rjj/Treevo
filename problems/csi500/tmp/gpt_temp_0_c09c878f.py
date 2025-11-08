import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Extract price columns
    open_price = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Multi-Timeframe Momentum Calculation
    # Intraday Momentum
    intraday_raw = (close - open_price) / (high - low + epsilon)
    intraday_direction = np.sign(close - open_price)
    intraday_strength = np.abs(intraday_raw)
    
    # Short-term Momentum (3-day)
    price_change_3d = close / close.shift(3) - 1
    avg_range_3d = (high - low).rolling(window=3, min_periods=1).mean()
    short_term_momentum = price_change_3d / (avg_range_3d + epsilon)
    
    # Medium-term Momentum (10-day)
    price_change_10d = close / close.shift(10) - 1
    avg_range_10d = (high - low).rolling(window=10, min_periods=1).mean()
    medium_term_momentum = price_change_10d / (avg_range_10d + epsilon)
    
    # Momentum Persistence Analysis
    def calculate_persistence(direction_series, decay_factor=0.95):
        persistence = pd.Series(0.0, index=direction_series.index)
        current_count = 0
        
        for i in range(1, len(direction_series)):
            if direction_series.iloc[i] == direction_series.iloc[i-1] and not np.isnan(direction_series.iloc[i]):
                current_count += 1
            else:
                current_count = 0
            
            persistence.iloc[i] = current_count * (decay_factor ** current_count)
        
        return persistence
    
    # Calculate persistence for each momentum type
    intraday_persistence = calculate_persistence(intraday_direction)
    short_term_direction = np.sign(short_term_momentum)
    short_term_persistence = calculate_persistence(short_term_direction)
    medium_term_direction = np.sign(medium_term_momentum)
    medium_term_persistence = calculate_persistence(medium_term_direction)
    
    # Persistence-weighted momentum strength
    intraday_persistent = intraday_strength * (1 + intraday_persistence)
    short_term_persistent = np.abs(short_term_momentum) * (1 + short_term_persistence)
    medium_term_persistent = np.abs(medium_term_momentum) * (1 + medium_term_persistence)
    
    # Volume Confirmation Signals
    volume_ratio = volume / volume.shift(1)
    volume_direction = np.sign(volume_ratio - 1)
    volume_strength = np.abs(volume_ratio - 1)
    
    # Volume persistence
    volume_persistence = calculate_persistence(volume_direction)
    
    # Volume-Price Alignment
    volume_price_alignment = (
        (volume_direction == intraday_direction).astype(float) * 0.4 +
        (volume_direction == short_term_direction).astype(float) * 0.3 +
        (volume_direction == medium_term_direction).astype(float) * 0.3
    )
    
    # Volume alignment strength
    volume_alignment_strength = volume_price_alignment * (1 + volume_persistence) * volume_strength
    
    # Final Factor Construction
    # Volatility-Normalized Momentum Blend with persistence weights
    momentum_weights = (
        intraday_persistent * 0.4 +
        short_term_persistent * 0.35 +
        medium_term_persistent * 0.25
    )
    
    # Directional momentum blend
    directional_momentum = (
        intraday_direction * intraday_persistent * 0.4 +
        short_term_direction * short_term_persistent * 0.35 +
        medium_term_direction * medium_term_persistent * 0.25
    )
    
    # Apply volume alignment enhancement
    volume_enhanced_momentum = directional_momentum * (1 + volume_alignment_strength)
    
    # Adaptive volatility adjustment
    combined_volatility = (avg_range_3d * 0.4 + avg_range_10d * 0.6)
    volatility_adjusted_factor = volume_enhanced_momentum / (combined_volatility + epsilon)
    
    # Final factor with momentum persistence and volume confirmation components
    final_factor = volatility_adjusted_factor * momentum_weights
    
    # Clean any infinite or NaN values
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return final_factor
