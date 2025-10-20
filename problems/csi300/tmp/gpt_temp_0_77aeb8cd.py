import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Horizon Price-Volume Convergence Factor
    Combines price and volume momentum across multiple time horizons with alignment analysis
    and persistence scoring to generate a predictive alpha factor.
    """
    df = data.copy()
    
    # Price Momentum Assessment
    df['price_momentum_60d'] = np.sign(df['close'] / df['close'].shift(60) - 1)
    df['price_momentum_20d'] = np.sign(df['close'] / df['close'].shift(20) - 1)
    df['price_momentum_5d'] = np.sign(df['close'] / df['close'].shift(5) - 1)
    
    # Volume Momentum Assessment
    df['volume_momentum_60d'] = np.sign(df['volume'] / df['volume'].shift(60) - 1)
    df['volume_momentum_20d'] = np.sign(df['volume'] / df['volume'].shift(20) - 1)
    df['volume_momentum_5d'] = np.sign(df['volume'] / df['volume'].shift(5) - 1)
    
    # Sign Alignment Analysis
    df['alignment_60d'] = df['price_momentum_60d'] * df['volume_momentum_60d']
    df['alignment_20d'] = df['price_momentum_20d'] * df['volume_momentum_20d']
    df['alignment_5d'] = df['price_momentum_5d'] * df['volume_momentum_5d']
    
    # Magnitude Weighting
    df['price_magnitude'] = (abs(df['price_momentum_60d']) + 
                            abs(df['price_momentum_20d']) + 
                            abs(df['price_momentum_5d'])) / 3
    df['volume_magnitude'] = (abs(df['volume_momentum_60d']) + 
                             abs(df['volume_momentum_20d']) + 
                             abs(df['volume_momentum_5d'])) / 3
    df['combined_magnitude'] = np.minimum(df['price_magnitude'], df['volume_magnitude'])
    
    # Trend Persistence Scoring
    # Price trend consistency
    price_signs = df[['price_momentum_60d', 'price_momentum_20d', 'price_momentum_5d']]
    df['price_persistence'] = (price_signs.nunique(axis=1) == 1).astype(int) / 3
    
    # Volume trend consistency
    volume_signs = df[['volume_momentum_60d', 'volume_momentum_20d', 'volume_momentum_5d']]
    df['volume_persistence'] = (volume_signs.nunique(axis=1) == 1).astype(int) / 3
    
    # Cross-asset persistence
    alignment_signs = df[['alignment_60d', 'alignment_20d', 'alignment_5d']]
    df['cross_persistence'] = (alignment_signs == 1).sum(axis=1) / 3
    
    # Factor Construction
    # Alignment strength
    df['alignment_strength'] = (df['alignment_60d'] * df['combined_magnitude'] + 
                               df['alignment_20d'] * df['combined_magnitude'] + 
                               df['alignment_5d'] * df['combined_magnitude'])
    
    # Persistence multiplier
    df['persistence_multiplier'] = (df['price_persistence'] + 
                                   df['volume_persistence'] + 
                                   df['cross_persistence']) / 3
    
    # Final factor
    factor = df['alignment_strength'] * df['persistence_multiplier']
    
    return factor
