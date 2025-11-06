import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Decay and Volume Acceleration Alpha Factor
    Combines momentum decay patterns with volume acceleration signals
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Price Momentum Decay Patterns
    # Short-term momentum decay rate
    data['momentum_decay_rate'] = (data['close'] - data['close'].shift(1)) / \
                                 (data['close'].shift(1) - data['close'].shift(3)).replace(0, np.nan)
    
    # Momentum inflection detection
    data['momentum_inflection'] = ((data['close'] - data['close'].shift(1)) * \
                                 (data['close'].shift(1) - data['close'].shift(2))) < 0
    
    # Momentum persistence (5-day count)
    momentum_sign = np.sign(data['close'] - data['close'].shift(1))
    momentum_persistence = []
    for i in range(len(data)):
        if i < 5:
            momentum_persistence.append(np.nan)
        else:
            window = momentum_sign.iloc[i-4:i+1]
            persistence_count = (window == window.iloc[-1]).sum()
            momentum_persistence.append(persistence_count)
    data['momentum_persistence'] = momentum_persistence
    
    # Multi-timeframe momentum divergence
    data['momentum_3d'] = np.sign(data['close'] - data['close'].shift(3))
    data['momentum_8d'] = np.sign(data['close'] - data['close'].shift(8))
    data['momentum_divergence'] = data['momentum_3d'] != data['momentum_8d']
    
    # Momentum magnitude ratio
    data['momentum_magnitude_ratio'] = abs(data['close'] - data['close'].shift(3)) / \
                                      abs(data['close'] - data['close'].shift(8)).replace(0, np.nan)
    
    # Volatility-adjusted momentum
    data['high_5d'] = data['high'].rolling(window=5, min_periods=5).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=5).min()
    data['momentum_vol_adj'] = (data['close'] - data['close'].shift(5)) / \
                              (data['high_5d'] - data['low_5d']).replace(0, np.nan)
    
    # Volatility expansion
    data['vol_expansion'] = (data['high'] - data['low']) / \
                           (data['high'].shift(5) - data['low'].shift(5)).replace(0, np.nan)
    
    # Momentum stability
    data['momentum_stability'] = data['close'].rolling(window=5, min_periods=5).std() / \
                                data['close'].rolling(window=5, min_periods=5).mean()
    
    # Volume Acceleration Patterns
    # Simple volume acceleration
    data['volume_accel_simple'] = (data['volume'] - data['volume'].shift(1)) / \
                                 data['volume'].shift(1).replace(0, np.nan)
    
    # Multi-period acceleration
    data['volume_accel_multi'] = (data['volume'] / data['volume'].shift(5)) - \
                                (data['volume'].shift(5) / data['volume'].shift(10))
    
    # Volume acceleration persistence
    volume_increase = (data['volume'] - data['volume'].shift(1)) > 0
    volume_persistence = []
    for i in range(len(data)):
        if i < 3:
            volume_persistence.append(np.nan)
        else:
            window = volume_increase.iloc[i-2:i+1]
            persistence_count = window.sum()
            volume_persistence.append(persistence_count)
    data['volume_accel_persistence'] = volume_persistence
    
    # Volume shock identification
    data['volume_shock_abs'] = data['volume'] / data['volume'].shift(1).replace(0, np.nan)
    data['volume_median_20d'] = data['volume'].rolling(window=20, min_periods=20).median()
    data['volume_shock_rel'] = data['volume'] / data['volume_median_20d']
    
    # Volume breakout
    data['volume_max_10d'] = data['volume'].rolling(window=10, min_periods=10).max()
    data['volume_breakout'] = data['volume'] > (1.5 * data['volume_max_10d'])
    
    # Volume regime analysis
    data['volume_trend'] = data['volume'] / data['volume'].shift(21)
    data['volume_volatility'] = data['volume'].rolling(window=5, min_periods=5).std() / \
                               data['volume'].rolling(window=5, min_periods=5).mean()
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=5).sum()
    
    # Volatility Context Integration
    # Daily range volatility
    data['daily_volatility'] = (data['high'] - data['low']) / data['close']
    
    # Multi-day volatility
    data['high_5d_range'] = data['high'].rolling(window=5, min_periods=5).max()
    data['low_5d_range'] = data['low'].rolling(window=5, min_periods=5).min()
    data['multi_day_volatility'] = (data['high_5d_range'] - data['low_5d_range']) / data['close']
    
    # Volatility trend
    data['volatility_trend'] = (data['high'] - data['low']) / \
                              (data['high'].shift(5) - data['low'].shift(5)).replace(0, np.nan)
    
    # Volatility-adjusted volume
    data['volume_accel_vol_adj'] = data['volume_accel_simple'] / data['daily_volatility'].replace(0, np.nan)
    data['volume_shock_vol_adj'] = data['volume_shock_abs'] / data['daily_volatility'].replace(0, np.nan)
    data['volume_persistence_vol'] = data['volume_trend'] * data['volatility_trend']
    
    # Volatility regime detection
    data['high_vol_period'] = (data['high'] - data['low']) > (1.5 * (data['high'].shift(5) - data['low'].shift(5)))
    data['low_vol_compression'] = (data['high'] - data['low']) < (0.7 * (data['high'].shift(5) - data['low'].shift(5)))
    
    # Momentum-Volume Interaction
    # Volume confirmation of momentum
    data['high_volume_momentum'] = data['momentum_decay_rate'] * data['volume_accel_simple']
    data['low_volume_decay'] = data['momentum_decay_rate'] * (1 / data['volume_accel_simple'].replace(0, np.nan))
    
    # Final Alpha Integration
    # Decay-Acceleration Combination
    
    # Momentum decay scoring
    decay_magnitude = -abs(data['momentum_decay_rate'])  # Negative for decay
    decay_timing = data['momentum_inflection'].astype(float) * -1  # Negative for inflection
    decay_reliability = data['momentum_persistence'] / 5  # Normalized persistence
    
    # Volume acceleration probability
    accel_timing = data['volume_accel_persistence'] / 3  # Normalized persistence
    accel_magnitude = data['volume_accel_vol_adj']
    accel_directionality = np.sign(data['volume_accel_simple']) * data['volume_shock_rel']
    
    # Combine decay and acceleration signals
    decay_score = (decay_magnitude * 0.4 + decay_timing * 0.3 + decay_reliability * 0.3)
    accel_prob = (accel_timing * 0.3 + accel_magnitude * 0.4 + accel_directionality * 0.3)
    
    # Final alpha with volatility adjustment
    volatility_weight = 1 / (1 + data['daily_volatility'])  # Lower weight in high volatility
    
    # Momentum Decay and Volume Acceleration Alpha
    alpha = decay_score * accel_prob * volatility_weight
    
    # Clean up and return
    alpha_series = alpha.replace([np.inf, -np.inf], np.nan)
    
    return alpha_series
