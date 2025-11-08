import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Momentum Divergence Alpha Factor
    Combines price and volume momentum across multiple timeframes with divergence pattern detection
    """
    df = data.copy()
    
    # Price Momentum Calculation
    df['price_momentum_3d'] = (df['close'] / df['close'].shift(3) - 1)
    df['price_momentum_8d'] = (df['close'] / df['close'].shift(8) - 1)
    
    # Volume Momentum Calculation
    df['volume_momentum_3d'] = (df['volume'] / df['volume'].shift(3) - 1)
    df['volume_momentum_8d'] = (df['volume'] / df['volume'].shift(8) - 1)
    
    # Divergence Pattern Detection
    df['divergence_3d'] = np.sign(df['price_momentum_3d']) != np.sign(df['volume_momentum_3d'])
    df['divergence_8d'] = np.sign(df['price_momentum_8d']) != np.sign(df['volume_momentum_8d'])
    
    # Pattern Classification
    df['bullish_divergence_3d'] = ((df['price_momentum_3d'] < 0) & (df['volume_momentum_3d'] > 0) & df['divergence_3d']).astype(int)
    df['bearish_divergence_3d'] = ((df['price_momentum_3d'] > 0) & (df['volume_momentum_3d'] < 0) & df['divergence_3d']).astype(int)
    df['bullish_divergence_8d'] = ((df['price_momentum_8d'] < 0) & (df['volume_momentum_8d'] > 0) & df['divergence_8d']).astype(int)
    df['bearish_divergence_8d'] = ((df['price_momentum_8d'] > 0) & (df['volume_momentum_8d'] < 0) & df['divergence_8d']).astype(int)
    
    # Divergence signal (bullish positive, bearish negative)
    df['divergence_signal_3d'] = df['bullish_divergence_3d'] - df['bearish_divergence_3d']
    df['divergence_signal_8d'] = df['bullish_divergence_8d'] - df['bearish_divergence_8d']
    
    # Persistence Analysis
    df['divergence_persistence_3d'] = 0
    df['divergence_persistence_8d'] = 0
    
    # Calculate consecutive divergence days
    for i in range(1, len(df)):
        if df['divergence_signal_3d'].iloc[i] != 0:
            if df['divergence_signal_3d'].iloc[i] == df['divergence_signal_3d'].iloc[i-1]:
                df['divergence_persistence_3d'].iloc[i] = df['divergence_persistence_3d'].iloc[i-1] + 1
            else:
                df['divergence_persistence_3d'].iloc[i] = 1
                
        if df['divergence_signal_8d'].iloc[i] != 0:
            if df['divergence_signal_8d'].iloc[i] == df['divergence_signal_8d'].iloc[i-1]:
                df['divergence_persistence_8d'].iloc[i] = df['divergence_persistence_8d'].iloc[i-1] + 1
            else:
                df['divergence_persistence_8d'].iloc[i] = 1
    
    # Cap persistence effect at reasonable limit (5 days)
    df['persistence_weight_3d'] = np.minimum(df['divergence_persistence_3d'], 5)
    df['persistence_weight_8d'] = np.minimum(df['divergence_persistence_8d'], 5)
    
    # Multi-timeframe Divergence Integration
    df['weighted_divergence_3d'] = df['divergence_signal_3d'] * df['persistence_weight_3d']
    df['weighted_divergence_8d'] = df['divergence_signal_8d'] * df['persistence_weight_8d']
    
    # Combined divergence signal
    df['combined_divergence'] = 0.6 * df['weighted_divergence_3d'] + 0.4 * df['weighted_divergence_8d']
    
    # Momentum Magnitude Scaling
    price_momentum_magnitude = np.abs(df['price_momentum_3d']) + np.abs(df['price_momentum_8d'])
    volume_momentum_magnitude = np.abs(df['volume_momentum_3d']) + np.abs(df['volume_momentum_8d'])
    
    # Final factor construction
    df['alpha_factor'] = (df['combined_divergence'] * 
                         price_momentum_magnitude * 
                         np.log1p(volume_momentum_magnitude))
    
    return df['alpha_factor']
