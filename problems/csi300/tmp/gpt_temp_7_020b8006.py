import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Stable Momentum Acceleration with Volume Confirmation alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Core Momentum Acceleration
    # Multi-Timeframe Momentum
    data['short_momentum'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['medium_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['long_momentum'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Acceleration Detection
    data['short_to_medium'] = data['medium_momentum'] - data['short_momentum']
    data['medium_to_long'] = data['long_momentum'] - data['medium_momentum']
    data['acceleration_alignment'] = np.sign(data['short_to_medium']) == np.sign(data['medium_to_long'])
    
    # Momentum Persistence
    # Direction consistency over 3 days
    data['medium_momentum_sign'] = np.sign(data['medium_momentum'])
    data['direction_consistency'] = data['medium_momentum_sign'].rolling(window=3).apply(
        lambda x: len(set(x)) == 1 if not x.isna().any() else 0
    )
    
    # Acceleration persistence
    data['acceleration_persistence'] = data['acceleration_alignment'].rolling(window=3).apply(
        lambda x: x.sum() if not x.isna().any() else 0
    )
    
    # Momentum quality
    data['momentum_quality'] = data['medium_momentum'] * (1 + data['direction_consistency'] / 5)
    
    # Volume Confirmation Engine
    # Volume Dynamics
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_trend'] = data['volume'] / data['volume'].shift(5)
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    
    # Price-Volume Alignment
    data['short_alignment'] = np.sign(data['volume_ratio'] - 1) == np.sign(data['short_momentum'])
    data['medium_alignment'] = np.sign(data['volume_trend'] - 1) == np.sign(data['medium_momentum'])
    data['momentum_alignment'] = np.sign(data['volume_momentum']) == np.sign(data['short_to_medium'])
    
    # Volume Confidence
    data['alignment_score'] = (data['short_alignment'].astype(int) + 
                              data['medium_alignment'].astype(int) + 
                              data['momentum_alignment'].astype(int))
    data['volume_strength'] = (data['volume_ratio'] + data['volume_trend'] + data['volume_momentum']) / 3
    data['volume_multiplier'] = 1 + (data['alignment_score'] * 0.15)
    
    # Volatility Stability Framework
    # Range Analysis
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['avg_5day_range'] = data['daily_range'].rolling(window=5).mean()
    data['range_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Volatility Stability
    data['range_stability'] = data['daily_range'].rolling(window=5).apply(
        lambda x: sum((x >= x.mean() * 0.85) & (x <= x.mean() * 1.15)) if not x.isna().any() else 0
    )
    
    # Volatility persistence
    data['range_pct_change'] = data['daily_range'].pct_change()
    data['volatility_persistence'] = data['range_pct_change'].rolling(window=5).apply(
        lambda x: sum(np.abs(x) <= 0.1) if not x.isna().any() else 0
    )
    
    data['stability_score'] = (data['range_stability'] + data['volatility_persistence']) / 10
    
    # Volatility Adjustment
    data['range_normalization'] = 1 / (data['daily_range'] + 0.0001)
    data['stability_multiplier'] = 1 + data['stability_score']
    data['volatility_confidence'] = np.minimum(1.0, data['stability_score'] * 1.5)
    
    # Signal Integration & Validation
    # Base Signal Construction
    data['accelerated_momentum'] = data['momentum_quality'] * (1 + data['short_to_medium'])
    data['volume_confirmed'] = data['accelerated_momentum'] * data['volume_multiplier']
    data['persistence_enhanced'] = data['volume_confirmed'] * (1 + data['acceleration_persistence'] / 8)
    
    # Confidence Filtering
    data['high_confidence'] = (data['alignment_score'] >= 2) & (data['volatility_confidence'] > 0.6)
    data['medium_confidence'] = (data['alignment_score'] >= 1) | (data['volatility_confidence'] > 0.3)
    data['valid_signal'] = data['medium_confidence']
    
    # Volatility Scaling
    data['range_adjusted'] = data['persistence_enhanced'] * data['range_normalization'] * data['valid_signal']
    data['stability_scaled'] = data['range_adjusted'] * data['stability_multiplier']
    data['final_factor'] = data['stability_scaled'] * 500
    
    # Clean up intermediate columns and return final factor
    factor = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor
