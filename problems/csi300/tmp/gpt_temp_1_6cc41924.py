import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Extraction
    # Price Momentum Components
    fast_price_momentum = (df['close'] - df['close'].shift(3)) / (
        df['high'].rolling(window=4).max() - df['low'].rolling(window=4).min() + 0.0001
    )
    
    medium_price_momentum = (df['close'] - df['close'].shift(8)) / (
        df['high'].rolling(window=9).max() - df['low'].rolling(window=9).min() + 0.0001
    )
    
    slow_price_momentum = (df['close'] - df['close'].shift(15)) / (
        df['high'].rolling(window=16).max() - df['low'].rolling(window=16).min() + 0.0001
    )
    
    # Volume Momentum Components
    fast_volume_momentum = (df['volume'] - df['volume'].shift(3)) / (
        df['volume'].rolling(window=4).apply(lambda x: np.max(np.abs(x))) + 0.0001
    )
    
    medium_volume_momentum = (df['volume'] - df['volume'].shift(8)) / (
        df['volume'].rolling(window=9).apply(lambda x: np.max(np.abs(x))) + 0.0001
    )
    
    slow_volume_momentum = (df['volume'] - df['volume'].shift(15)) / (
        df['volume'].rolling(window=16).apply(lambda x: np.max(np.abs(x))) + 0.0001
    )
    
    # Volatility Regime Classification
    current_volatility_proxy = (df['high'] - df['low']) / df['close']
    historical_volatility_reference = (
        df['high'].rolling(window=11).max() - df['low'].rolling(window=11).min()
    ) / df['close'].shift(5)
    regime_indicator = current_volatility_proxy / (historical_volatility_reference + 0.0001)
    
    # Adaptive Weighting Scheme
    low_volatility_weight = np.maximum(0, 1 - regime_indicator)
    normal_volatility_weight = np.exp(-np.abs(regime_indicator - 1))
    high_volatility_weight = np.maximum(0, regime_indicator - 1)
    
    # Momentum Divergence Construction
    fast_divergence = fast_price_momentum - fast_volume_momentum
    medium_divergence = medium_price_momentum - medium_volume_momentum
    slow_divergence = slow_price_momentum - slow_volume_momentum
    
    # Regime-Adaptive Blending
    low_volatility_component = slow_divergence * low_volatility_weight
    normal_volatility_component = medium_divergence * normal_volatility_weight
    high_volatility_component = fast_divergence * high_volatility_weight
    
    # Final Alpha Output
    alpha = low_volatility_component + normal_volatility_component + high_volatility_component
    
    return alpha
