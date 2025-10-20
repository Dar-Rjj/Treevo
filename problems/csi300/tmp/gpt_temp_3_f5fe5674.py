import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Horizon Price-Volume Convergence Factor
    Combines price and volume momentum across three time horizons with convergence assessment
    """
    close = df['close']
    volume = df['volume']
    
    # Calculate EWMA for different horizons
    ewma_60_close = close.ewm(span=60, adjust=False).mean()
    ewma_60_volume = volume.ewm(span=60, adjust=False).mean()
    ewma_20_close = close.ewm(span=20, adjust=False).mean()
    ewma_20_volume = volume.ewm(span=20, adjust=False).mean()
    ewma_5_close = close.ewm(span=5, adjust=False).mean()
    ewma_5_volume = volume.ewm(span=5, adjust=False).mean()
    
    # Long-term momentum (60-day)
    price_momentum_60 = (close / ewma_60_close) - 1
    volume_momentum_60 = (volume / ewma_60_volume) - 1
    
    # Medium-term momentum (20-day)
    price_momentum_20 = (close / ewma_20_close) - 1
    volume_momentum_20 = (volume / ewma_20_volume) - 1
    
    # Short-term momentum (5-day)
    price_momentum_5 = (close / ewma_5_close) - 1
    volume_momentum_5 = (volume / ewma_5_volume) - 1
    
    # Momentum magnitude scaling
    long_term_magnitude = np.abs(price_momentum_60) * np.abs(volume_momentum_60)
    medium_term_magnitude = np.abs(price_momentum_20) * np.abs(volume_momentum_20)
    short_term_magnitude = np.abs(price_momentum_5) * np.abs(volume_momentum_5)
    
    # Convergence assessment
    long_term_alignment = np.sign(price_momentum_60) * np.sign(volume_momentum_60)
    medium_term_alignment = np.sign(price_momentum_20) * np.sign(volume_momentum_20)
    short_term_alignment = np.sign(price_momentum_5) * np.sign(volume_momentum_5)
    
    # Horizon consistency score
    alignment_matrix = pd.DataFrame({
        'long': long_term_alignment,
        'medium': medium_term_alignment,
        'short': short_term_alignment
    })
    
    def calculate_convergence_score(row):
        positive_count = (row > 0).sum()
        negative_count = (row < 0).sum()
        zero_count = (row == 0).sum()
        
        if positive_count == 3:
            return 3.0
        elif negative_count == 3:
            return 3.0
        elif positive_count == 2 and negative_count == 0:
            return 2.0
        elif negative_count == 2 and positive_count == 0:
            return 2.0
        elif positive_count == 1 and negative_count == 1:
            return 1.0
        else:
            return 0.5
    
    convergence_multiplier = alignment_matrix.apply(calculate_convergence_score, axis=1)
    
    # Base blended signal
    long_term_component = price_momentum_60 * volume_momentum_60
    medium_term_component = price_momentum_20 * volume_momentum_20
    short_term_component = price_momentum_5 * volume_momentum_5
    
    base_blended_signal = (
        0.5 * long_term_component + 
        0.3 * medium_term_component + 
        0.2 * short_term_component
    )
    
    # Momentum magnitude scale
    momentum_magnitude_scale = (
        0.5 * long_term_magnitude + 
        0.3 * medium_term_magnitude + 
        0.2 * short_term_magnitude
    )
    
    # Final factor construction
    directional_component = base_blended_signal * convergence_multiplier
    final_factor = directional_component * momentum_magnitude_scale
    
    return final_factor
