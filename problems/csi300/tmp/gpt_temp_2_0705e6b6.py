import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Dimensional Momentum Fragmentation Factor
    Analyzes momentum consistency across price, volume, and range dimensions
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Components
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['close_momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Volume Momentum Components
    data['volume_change_momentum'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1).replace(0, np.nan)
    data['volume_price_alignment'] = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['close'] - data['close'].shift(1))
    data['volume_acceleration'] = (data['volume'] - data['volume'].shift(1)) - (data['volume'].shift(1) - data['volume'].shift(2))
    
    # Range Momentum Components
    data['range_expansion_momentum'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['range_position_current'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['range_position_momentum'] = data['range_position_current'] - data['range_position_current'].shift(1)
    data['range_utilization_current'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['range_utilization_momentum'] = data['range_utilization_current'] - data['range_utilization_current'].shift(1)
    
    # Fragmentation Detection - Component Divergence Analysis
    data['price_volume_divergence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['intraday_range_divergence'] = np.sign(data['close'] - data['open']) * np.sign((data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1)))
    data['gap_close_divergence'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Market State Integration
    data['absolute_volatility'] = data['high'] - data['low']
    data['relative_volatility'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['volatility_regime'] = data['absolute_volatility'].rolling(window=10, min_periods=5).rank(pct=True)
    
    data['volume_level'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_stability'] = data['volume'] / data['volume'].shift(1).replace(0, np.nan)
    data['price_impact'] = np.abs(data['close'] - data['open']) * data['volume']
    
    # Trend Context
    data['short_term_trend'] = np.sign(data['close'] - data['close'].shift(3))
    data['medium_term_trend'] = np.sign(data['close'] - data['close'].shift(7))
    high_7d = data['high'].rolling(window=7, min_periods=5).max()
    low_7d = data['low'].rolling(window=7, min_periods=5).min()
    data['trend_strength'] = np.abs(data['close'] - data['close'].shift(7)) / (high_7d - low_7d).replace(0, np.nan)
    
    # Momentum Consistency Scoring
    momentum_components = [
        'intraday_momentum', 'gap_momentum', 'close_momentum',
        'volume_change_momentum', 'volume_price_alignment', 'volume_acceleration',
        'range_expansion_momentum', 'range_position_momentum', 'range_utilization_momentum'
    ]
    
    # Calculate directional consistency
    directional_signs = pd.DataFrame()
    for col in momentum_components:
        directional_signs[col] = np.sign(data[col])
    
    # Count consistent directional moves
    data['directional_consistency_count'] = directional_signs.apply(
        lambda x: x.value_counts().max() if len(x.value_counts()) > 0 else 0, axis=1
    )
    
    # Magnitude consistency (coefficient of variation)
    momentum_magnitudes = data[momentum_components].abs()
    data['magnitude_consistency'] = 1 / (momentum_magnitudes.std(axis=1) / momentum_magnitudes.mean(axis=1)).replace(0, np.nan)
    
    # Fragmentation Pattern Recognition
    data['convergence_pattern'] = (
        (data['price_volume_divergence'] > 0) & 
        (data['intraday_range_divergence'] > 0) & 
        (data['gap_close_divergence'] > 0)
    ).astype(int)
    
    data['divergence_pattern'] = (
        (data['price_volume_divergence'] < 0) | 
        (data['intraday_range_divergence'] < 0) | 
        (data['gap_close_divergence'] < 0)
    ).astype(int)
    
    # Factor Construction - Fragmentation Score
    # Higher fragmentation = more conflicting signals
    fragmentation_components = [
        'price_volume_divergence', 'intraday_range_divergence', 'gap_close_divergence'
    ]
    
    # Calculate fragmentation score (negative values indicate fragmentation)
    data['raw_fragmentation'] = (
        data['directional_consistency_count'] * -1 +  # Lower consistency = higher fragmentation
        data['divergence_pattern'] * 2 +  # More divergence patterns = higher fragmentation
        (1 - data['convergence_pattern']) * 1  # Less convergence = higher fragmentation
    )
    
    # Market State Adjustment
    volatility_weight = 1 / (data['volatility_regime'] + 0.1)  # Higher weight in low volatility
    liquidity_weight = np.where(data['volume_level'] > 1, 1, 0.5)  # Higher weight with good liquidity
    
    # Final factor construction
    data['fragmentation_factor'] = (
        data['raw_fragmentation'] * 
        volatility_weight * 
        liquidity_weight * 
        (1 + data['trend_strength'].fillna(0))  # Amplify in strong trends
    )
    
    # Risk adjustment and normalization
    factor_volatility = data['fragmentation_factor'].rolling(window=20, min_periods=10).std()
    data['final_factor'] = data['fragmentation_factor'] / factor_volatility.replace(0, np.nan)
    
    # Apply trend alignment
    data['final_factor'] = data['final_factor'] * data['medium_term_trend'].replace(0, 1)
    
    return data['final_factor']
