import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum with Volume-Price Synergy factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily range
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # Multi-Timeframe Volatility Regime Classification
    data['vol_short'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['vol_medium'] = data['daily_range'].rolling(window=20, min_periods=10).median()
    
    # Regime classification with hysteresis
    high_threshold = 1.3 * data['vol_medium']
    low_threshold = 0.8 * data['vol_medium']
    
    # Initialize regime column
    data['regime'] = 1  # Normal volatility by default
    
    # Apply regime classification with hysteresis
    for i in range(1, len(data)):
        prev_regime = data['regime'].iloc[i-1]
        current_vol_short = data['vol_short'].iloc[i]
        current_vol_medium = data['vol_medium'].iloc[i]
        
        if prev_regime == 2:  # Previous was high volatility
            if current_vol_short < 1.1 * current_vol_medium:
                data.loc[data.index[i], 'regime'] = 1  # Transition to normal
            else:
                data.loc[data.index[i], 'regime'] = 2  # Stay high
        elif prev_regime == 0:  # Previous was low volatility
            if current_vol_short > 0.9 * current_vol_medium:
                data.loc[data.index[i], 'regime'] = 1  # Transition to normal
            else:
                data.loc[data.index[i], 'regime'] = 0  # Stay low
        else:  # Previous was normal
            if current_vol_short > high_threshold.iloc[i]:
                data.loc[data.index[i], 'regime'] = 2  # High volatility
            elif current_vol_short < low_threshold.iloc[i]:
                data.loc[data.index[i], 'regime'] = 0  # Low volatility
            else:
                data.loc[data.index[i], 'regime'] = 1  # Stay normal
    
    # Regime stability assessment
    data['regime_duration'] = 0
    for i in range(1, len(data)):
        if data['regime'].iloc[i] == data['regime'].iloc[i-1]:
            data.loc[data.index[i], 'regime_duration'] = data['regime_duration'].iloc[i-1] + 1
    
    data['regime_transition_prob'] = 1.0 / (1.0 + np.exp(-0.1 * (data['regime_duration'] - 5)))
    
    # Multi-period momentum calculation
    data['momentum_3d'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['momentum_6d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_8d'] = (data['close'] - data['close'].shift(7)) / data['close'].shift(7)
    
    # Momentum acceleration
    data['accel_3d'] = data['momentum_3d'] - data['momentum_3d'].shift(1)
    data['accel_6d'] = data['momentum_6d'] - data['momentum_6d'].shift(1)
    data['accel_8d'] = data['momentum_8d'] - data['momentum_8d'].shift(1)
    
    # Regime-specific momentum acceleration
    data['regime_momentum'] = 0.0
    high_vol_mask = data['regime'] == 2
    low_vol_mask = data['regime'] == 0
    normal_vol_mask = data['regime'] == 1
    
    data.loc[high_vol_mask, 'regime_momentum'] = data.loc[high_vol_mask, 'accel_3d']
    data.loc[low_vol_mask, 'regime_momentum'] = data.loc[low_vol_mask, 'accel_8d']
    data.loc[normal_vol_mask, 'regime_momentum'] = data.loc[normal_vol_mask, 'accel_6d']
    
    # Momentum quality assessment
    data['momentum_direction_agreement'] = (
        (np.sign(data['momentum_3d']) == np.sign(data['momentum_6d'])).astype(int) +
        (np.sign(data['momentum_3d']) == np.sign(data['momentum_8d'])).astype(int) +
        (np.sign(data['momentum_6d']) == np.sign(data['momentum_8d'])).astype(int)
    ) / 3.0
    
    data['momentum_persistence'] = (
        (data['momentum_3d'].rolling(window=3, min_periods=2).apply(lambda x: len(set(np.sign(x))) == 1, raw=False)).astype(float) +
        (data['momentum_6d'].rolling(window=3, min_periods=2).apply(lambda x: len(set(np.sign(x))) == 1, raw=False)).astype(float) +
        (data['momentum_8d'].rolling(window=3, min_periods=2).apply(lambda x: len(set(np.sign(x))) == 1, raw=False)).astype(float)
    ) / 3.0
    
    data['momentum_quality'] = (data['momentum_direction_agreement'] + data['momentum_persistence']) / 2.0
    
    # Volume-Price Synergy Confirmation
    # Volume-range efficiency
    data['vw_range'] = (data['daily_range'] * data['volume']).rolling(window=5, min_periods=3).sum() / data['volume'].rolling(window=5, min_periods=3).sum()
    data['avg_range'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['volume_range_efficiency'] = data['vw_range'] / (data['avg_range'] + 1e-8)
    
    # Price-volume convergence strength
    data['vwap'] = (data['close'] * data['volume']).rolling(window=5, min_periods=3).sum() / data['volume'].rolling(window=5, min_periods=3).sum()
    data['vwap_slope'] = data['vwap'].diff(3) / data['vwap'].shift(3)
    data['price_slope'] = data['close'].diff(3) / data['close'].shift(3)
    data['price_volume_alignment'] = np.sign(data['vwap_slope']) == np.sign(data['price_slope'])
    data['convergence_magnitude'] = np.abs(data['vwap_slope'] - data['price_slope'])
    
    # Volume flow impact assessment
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_surge'] = np.where(data['volume_ratio'] > 1.5, data['volume_ratio'] ** 0.5, 1.0)
    
    # Volume efficiency in driving price momentum
    data['volume_efficiency'] = data['momentum_3d'].abs() / (data['volume_ratio'] + 1e-8)
    
    # Integrated Signal Generation
    # Base signal: regime-adaptive momentum
    base_signal = data['regime_momentum']
    
    # Apply momentum quality as confidence multiplier
    quality_adjusted = base_signal * data['momentum_quality']
    
    # Scale by volume surge impact
    volume_scaled = quality_adjusted * data['volume_surge']
    
    # Adjust for regime transition probability
    regime_adjusted = volume_scaled * (1.0 - data['regime_transition_prob'])
    
    # Apply persistence filter for signal stability
    signal_persistence = (regime_adjusted.rolling(window=3, min_periods=2)
                         .apply(lambda x: x.mean() if len(set(np.sign(x))) == 1 else x.iloc[-1] * 0.5, raw=False))
    
    # Final factor with volume efficiency adjustment
    final_factor = signal_persistence * (1.0 + data['volume_efficiency'].clip(-0.5, 0.5))
    
    return final_factor
