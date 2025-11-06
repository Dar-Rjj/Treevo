import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multi-timeframe momentum, volume confirmation, 
    volatility adjustment, and regime-adaptive components.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum
    data['return_1d'] = data['close'] / data['close'].shift(1) - 1
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Momentum Quality
    data['return_consistency'] = data['return_1d'].rolling(3).apply(lambda x: (x > 0).sum())
    data['momentum_acceleration'] = data['return_3d'] - data['return_5d']
    data['momentum_persistence'] = data['return_3d'].rolling(3).apply(lambda x: (x > 0).sum())
    
    # Price Range Momentum
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_momentum'] = data['daily_range'] / data['daily_range'].shift(1) - 1
    data['range_adjusted_momentum'] = data['return_3d'] / (data['daily_range'] + 0.001)
    
    # Volume Confirmation Signals
    data['volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_trend'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_acceleration'] = data['volume_change'] - data['volume_change'].shift(1)
    
    data['volume_confirmed_return'] = data['return_3d'] * data['volume_trend']
    data['strong_volume_move'] = (data['volume_change'] > 0.5) * data['return_3d']
    data['volume_divergence'] = np.sign(data['return_3d']) * np.sign(data['volume_trend'])
    
    data['volume_momentum'] = data['volume_change'].rolling(3).apply(lambda x: (x > 0).sum())
    data['sustained_volume'] = data['volume_momentum'] * data['volume_trend']
    data['volume_regime'] = data['volume_trend'].rolling(3).apply(lambda x: (x > 0).sum())
    
    # Volatility-Adjusted Factors
    data['volatility_3d'] = data['return_1d'].rolling(3).std()
    data['volatility_5d'] = data['return_1d'].rolling(5).std()
    data['volatility_momentum'] = data['volatility_3d'] / data['volatility_5d'].shift(1) - 1
    
    data['volatility_scaled_momentum'] = data['return_3d'] / (data['volatility_3d'] + 0.001)
    data['clean_momentum'] = data['return_5d'] / (data['volatility_5d'] + 0.001)
    data['range_volatility_ratio'] = data['daily_range'] / (data['volatility_3d'] + 0.001)
    
    data['low_volatility_regime'] = (data['volatility_momentum'] < -0.1).astype(int)
    data['high_volatility_regime'] = (data['volatility_momentum'] > 0.1).astype(int)
    data['stable_volatility'] = (data['volatility_momentum'].abs() <= 0.1).astype(int)
    
    # Advanced Composite Factors
    data['volume_weighted_clean_momentum'] = data['clean_momentum'] * data['volume_trend']
    data['low_vol_volume_confirmation'] = data['low_volatility_regime'] * data['volume_confirmed_return']
    data['high_vol_momentum_quality'] = data['high_volatility_regime'] * data['momentum_persistence']
    
    data['consistent_volatility_scaled'] = data['volatility_scaled_momentum'] * data['return_consistency']
    data['accelerated_volume_momentum'] = data['momentum_acceleration'] * data['volume_confirmed_return']
    data['range_adjusted_volume_trend'] = data['range_adjusted_momentum'] * data['volume_trend']
    
    data['stable_regime_momentum'] = data['stable_volatility'] * data['clean_momentum']
    data['volume_persistent_acceleration'] = data['volume_regime'] * data['momentum_acceleration']
    data['multi_dimensional_quality'] = data['return_consistency'] * data['volume_divergence'] * data['volatility_scaled_momentum']
    
    # Final Alpha Output - weighted combination of core components
    alpha = (
        0.4 * data['volume_weighted_clean_momentum'] +
        0.25 * data['consistent_volatility_scaled'] +
        0.2 * data['accelerated_volume_momentum'] +
        0.15 * data['stable_regime_momentum']
    )
    
    return alpha
