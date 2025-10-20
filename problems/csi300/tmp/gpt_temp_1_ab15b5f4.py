import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on multi-horizon momentum integration with volume-price alignment.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    pd.Series: Alpha factor values indexed by date
    """
    # Calculate momentum components
    M15 = (df['close'] / df['close'].shift(15)) - 1
    M45 = (df['close'] / df['close'].shift(45)) - 1
    M90 = (df['close'] / df['close'].shift(90)) - 1
    
    # Calculate volume momentum components
    V15 = (df['volume'] / df['volume'].shift(15)) - 1
    V45 = (df['volume'] / df['volume'].shift(45)) - 1
    V90 = (df['volume'] / df['volume'].shift(90)) - 1
    
    # Weighted momentum base
    w15, w45, w90 = 0.6, 0.3, 0.1
    WM = w15 * M15 + w45 * M45 + w90 * M90
    
    # Volume-price alignment scoring
    Score15 = np.sign(M15) * np.sign(V15)
    Score45 = np.sign(M45) * np.sign(V45)
    Score90 = np.sign(M90) * np.sign(V90)
    volume_alignment_score = Score15 + Score45 + Score90
    
    # Momentum magnitude scaling
    avg_magnitude = (np.abs(M15) + np.abs(M45) + np.abs(M90)) / 3
    scale_factor = 1 + avg_magnitude
    
    # Directional convergence multiplier
    momentum_signs = [np.sign(M15), np.sign(M45), np.sign(M90)]
    unique_signs = len(set([x for x in momentum_signs if not np.isnan(x)]))
    
    if unique_signs == 1:  # All same sign
        mult_dir = 2.0
    elif unique_signs == 2:  # Partial convergence
        mult_dir = 1.5
    else:  # Divergence
        mult_dir = 0.5
    
    # Volume alignment multiplier
    if volume_alignment_score >= 2:
        mult_vol = 1.8
    elif volume_alignment_score == 1:
        mult_vol = 1.3
    elif volume_alignment_score == 0:
        mult_vol = 1.0
    elif volume_alignment_score == -1:
        mult_vol = 0.7
    else:  # score <= -2
        mult_vol = 0.3
    
    # Final alpha calculation
    alpha = WM * mult_dir * mult_vol * scale_factor
    
    return alpha
