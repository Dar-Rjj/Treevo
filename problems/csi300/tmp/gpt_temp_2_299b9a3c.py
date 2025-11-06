import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Acceleration with Volume-Price Alignment alpha factor
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
        Index should be datetime
    
    Returns:
    pd.Series: Alpha factor values indexed by date
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    data['momentum_1d'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['momentum_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Acceleration Gradient Analysis
    data['accel_short'] = data['momentum_3d'] - data['momentum_1d']
    data['accel_medium'] = data['momentum_5d'] - data['momentum_3d']
    data['accel_consistency'] = np.sign(data['accel_short']) * np.sign(data['accel_medium'])
    
    # Momentum Persistence Scoring
    momentum_direction = np.sign(data['momentum_3d'])
    persistence_count = momentum_direction.groupby(
        (momentum_direction != momentum_direction.shift(1)).cumsum()
    ).cumcount() + 1
    
    data['persistence_days'] = persistence_count
    data['persistence_weight'] = 0.0
    
    for i in range(len(data)):
        if data['persistence_days'].iloc[i] >= 1:
            days = min(data['persistence_days'].iloc[i], 10)  # Cap at 10 days
            data.loc[data.index[i], 'persistence_weight'] = sum(0.85 ** j for j in range(days))
    
    data['momentum_persistence_adj'] = data['momentum_3d'] * data['persistence_weight']
    
    # Volume-Price Alignment Engine
    data['volume_trend'] = data['volume'] / data['volume'].shift(1)
    data['volume_accel'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    data['volume_direction_align'] = np.sign(data['volume_trend']) * np.sign(data['momentum_3d'])
    
    # Price Range Efficiency
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['closing_strength'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['range_direction_align'] = np.sign(data['range_efficiency']) * np.sign(data['momentum_3d'])
    
    # Multi-Dimensional Confidence Score
    data['volume_confidence'] = np.where(data['volume_direction_align'] > 0, 1.0, 0.3)
    data['price_confidence'] = np.where(data['range_direction_align'] > 0, 1.0, 0.5)
    data['combined_confidence'] = data['volume_confidence'] * data['price_confidence']
    
    # Volatility-Scaled Signal Construction
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_5d_std'] = data['daily_range'].rolling(window=5).std()
    data['range_stability'] = 1 / (data['range_5d_std'] + 0.0001)
    data['volatility_regime'] = data['daily_range'] / data['daily_range'].rolling(window=20).mean()
    
    # Adaptive Scaling Framework
    data['base_signal'] = data['momentum_persistence_adj'] * data['combined_confidence']
    data['volatility_adj_signal'] = data['base_signal'] / (data['daily_range'] + 1e-8)
    data['stability_enhanced_signal'] = data['volatility_adj_signal'] * data['range_stability']
    
    # Signal Quality Filtering
    data['filtered_signal'] = data['stability_enhanced_signal']
    filter_mask = (data['persistence_days'] >= 2) & (data['accel_consistency'] > 0)
    data.loc[~filter_mask, 'filtered_signal'] = 0
    
    # Multi-Timeframe Signal Integration
    short_term_signal = data['momentum_1d'] * data['volume_confidence']
    medium_term_signal = data['filtered_signal']
    acceleration_signal = data['accel_short'] * data['combined_confidence']
    
    short_term_contrib = 0.3 * short_term_signal
    medium_term_contrib = 0.5 * medium_term_signal
    acceleration_contrib = 0.2 * acceleration_signal
    
    # Final Alpha Factor Output
    integrated_signal = short_term_contrib + medium_term_contrib + acceleration_contrib
    directional_consistency = np.sign(data['momentum_3d'])
    
    final_alpha = integrated_signal * directional_consistency
    
    return final_alpha
