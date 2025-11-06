import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Intraday Momentum Persistence factor
    """
    data = df.copy()
    
    # Calculate Intraday Momentum Persistence
    # Opening Gap Momentum
    data['overnight_return'] = data['open'] / data['close'].shift(1) - 1
    
    # Gap momentum persistence - track consecutive same-direction gaps
    data['gap_direction'] = np.sign(data['overnight_return'])
    data['gap_persistence'] = 0
    for i in range(1, len(data)):
        if data['gap_direction'].iloc[i] == data['gap_direction'].iloc[i-1]:
            data['gap_persistence'].iloc[i] = data['gap_persistence'].iloc[i-1] + 1
        else:
            data['gap_persistence'].iloc[i] = 0
    
    # Intraday Range Efficiency
    data['intraday_range'] = data['high'] - data['low']
    data['gap_magnitude'] = abs(data['overnight_return'])
    data['range_efficiency'] = data['intraday_range'] / (data['open'] * data['gap_magnitude'].replace(0, 0.001))
    
    # Momentum Continuation
    data['intraday_return'] = data['close'] / data['open'] - 1
    data['directional_consistency'] = np.where(
        np.sign(data['overnight_return']) == np.sign(data['intraday_return']), 1, -1
    )
    
    # Calculate persistence score
    data['momentum_persistence'] = (
        data['gap_persistence'] * data['range_efficiency'] * data['directional_consistency']
    )
    
    # Calculate Price-Volume Divergence
    # Directional Volume Alignment
    data['price_return'] = data['close'] / data['close'].shift(1) - 1
    data['volume_ma'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    
    # Identify Divergence Patterns
    data['price_volume_alignment'] = np.where(
        (data['price_return'] > 0) & (data['volume_ratio'] < 1), -1,  # Rising price, declining volume
        np.where(
            (data['price_return'] < 0) & (data['volume_ratio'] > 1), -1,  # Falling price, increasing volume
            1  # Aligned movement
        )
    )
    
    # Divergence magnitude
    data['divergence_magnitude'] = abs(data['price_return']) * (1 - data['volume_ratio'])
    data['divergence_strength'] = data['price_volume_alignment'] * data['divergence_magnitude']
    
    # Divergence Persistence
    data['divergence_persistence'] = 0
    for i in range(1, len(data)):
        if data['price_volume_alignment'].iloc[i] == data['price_volume_alignment'].iloc[i-1]:
            data['divergence_persistence'].iloc[i] = data['divergence_persistence'].iloc[i-1] + 1
        else:
            data['divergence_persistence'].iloc[i] = 0
    
    # Calculate Market Regime Sensitivity
    # Simple volatility-based regime classification
    data['volatility'] = data['close'].pct_change().rolling(window=20, min_periods=1).std()
    data['regime'] = np.where(data['volatility'] > data['volatility'].rolling(window=50, min_periods=1).median(), 1, 0)
    
    # Regime-adaptive parameters
    data['regime_weight'] = np.where(data['regime'] == 1, 0.7, 0.3)  # Higher weight to divergence in high vol
    
    # Combine with Dynamic Interaction
    # Core divergence factor
    data['core_divergence'] = data['momentum_persistence'] * data['divergence_strength']
    
    # Regime-adaptive blending
    data['regime_adjusted_divergence'] = (
        data['regime_weight'] * data['divergence_strength'] + 
        (1 - data['regime_weight']) * data['momentum_persistence']
    )
    
    # Final alpha factor
    data['alpha_factor'] = (
        data['core_divergence'] * data['regime_adjusted_divergence'] * 
        data['divergence_persistence']
    )
    
    # Normalize and clean
    data['alpha_factor'] = data['alpha_factor'].replace([np.inf, -np.inf], np.nan)
    data['alpha_factor'] = (data['alpha_factor'] - data['alpha_factor'].rolling(window=20, min_periods=1).mean()) / data['alpha_factor'].rolling(window=20, min_periods=1).std()
    
    return data['alpha_factor']
