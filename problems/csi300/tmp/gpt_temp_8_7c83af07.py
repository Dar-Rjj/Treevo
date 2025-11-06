import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Elasticity with Volume Confirmation factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Timeframe Momentum Divergence
    # Calculate momentum across different timeframes
    data['momentum_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['momentum_8d'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    data['momentum_21d'] = (data['close'] - data['close'].shift(21)) / data['close'].shift(21)
    
    # Assess momentum divergence patterns
    bullish_div = (data['momentum_3d'] > data['momentum_8d']) & (data['momentum_8d'] > data['momentum_21d'])
    bearish_div = (data['momentum_3d'] < data['momentum_8d']) & (data['momentum_8d'] < data['momentum_21d'])
    convergence = ((data['momentum_3d'] > 0) & (data['momentum_8d'] > 0) & (data['momentum_21d'] > 0)) | \
                  ((data['momentum_3d'] < 0) & (data['momentum_8d'] < 0) & (data['momentum_21d'] < 0))
    
    # Momentum divergence score
    data['momentum_div_score'] = 0
    data.loc[bullish_div, 'momentum_div_score'] = 1
    data.loc[bearish_div, 'momentum_div_score'] = -1
    data.loc[convergence, 'momentum_div_score'] = 0.5 * np.sign(data['momentum_3d'])
    
    # 2. Price Elasticity Characteristics
    # Calculate daily price stretch
    data['price_stretch'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Calculate elasticity momentum (3-day change in elasticity)
    data['elasticity_momentum'] = data['price_stretch'] / data['price_stretch'].shift(3) - 1
    
    # Elasticity classification
    high_elasticity = data['price_stretch'] > data['price_stretch'].rolling(window=10).mean()
    medium_elasticity = (data['price_stretch'] <= data['price_stretch'].rolling(window=10).mean()) & \
                       (data['price_stretch'] > data['price_stretch'].rolling(window=10).quantile(0.3))
    low_elasticity = data['price_stretch'] <= data['price_stretch'].rolling(window=10).quantile(0.3)
    
    # Elasticity-momentum alignment score
    data['elasticity_score'] = 0
    data.loc[high_elasticity & (data['momentum_div_score'] > 0), 'elasticity_score'] = 1.5
    data.loc[high_elasticity & (data['momentum_div_score'] < 0), 'elasticity_score'] = -1.5
    data.loc[medium_elasticity & (data['momentum_div_score'] > 0), 'elasticity_score'] = 1.0
    data.loc[medium_elasticity & (data['momentum_div_score'] < 0), 'elasticity_score'] = -1.0
    data.loc[low_elasticity & convergence, 'elasticity_score'] = 0.2 * np.sign(data['momentum_3d'])
    
    # 3. Volume Confirmation Strength
    # Calculate volume-momentum alignment
    positive_momentum_volume = data['volume'].where(data['momentum_3d'] > 0, 0)
    negative_momentum_volume = data['volume'].where(data['momentum_3d'] < 0, 0)
    
    # Volume confirmation ratio
    avg_positive_volume = positive_momentum_volume.rolling(window=5).mean()
    avg_negative_volume = negative_momentum_volume.rolling(window=5).mean()
    data['volume_confirmation_ratio'] = np.where(
        data['momentum_3d'] > 0,
        data['volume'] / avg_positive_volume,
        np.where(data['momentum_3d'] < 0,
                data['volume'] / avg_negative_volume,
                1)
    )
    
    # Volume persistence (3-day consistency)
    volume_trend = data['volume'].rolling(window=3).apply(
        lambda x: 1 if (x.diff().dropna() > 0).all() else (-1 if (x.diff().dropna() < 0).all() else 0),
        raw=False
    )
    
    # Volume efficiency (price move per unit volume)
    data['volume_efficiency'] = abs(data['close'] - data['close'].shift(1)) / (data['volume'] + 1e-8)
    avg_volume_efficiency = data['volume_efficiency'].rolling(window=10).mean()
    data['volume_efficiency_score'] = data['volume_efficiency'] / avg_volume_efficiency
    
    # Volume confirmation score
    data['volume_score'] = (
        np.clip(data['volume_confirmation_ratio'], 0.5, 2.0) * 
        np.where(volume_trend == np.sign(data['momentum_3d']), 1.2, 0.8) *
        np.clip(data['volume_efficiency_score'], 0.7, 1.5)
    )
    
    # 4. Generate Final Factor
    # Combine momentum divergence, elasticity, and volume confirmation
    data['momentum_elasticity_volume_factor'] = (
        data['momentum_div_score'] * 
        data['elasticity_score'] * 
        data['volume_score']
    )
    
    # Apply smoothing and normalization
    factor = data['momentum_elasticity_volume_factor'].rolling(window=3).mean()
    factor = (factor - factor.rolling(window=21).mean()) / factor.rolling(window=21).std()
    
    return factor
