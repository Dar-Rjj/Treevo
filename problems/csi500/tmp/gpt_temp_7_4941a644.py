import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Acceleration Diffusion with Microstructure Anchoring factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Acceleration Diffusion Component
    # Calculate price momentum changes
    data['price_ratio_1'] = data['close'] / data['close'].shift(1)
    data['price_ratio_2'] = data['close'].shift(1) / data['close'].shift(2)
    data['price_ratio_3'] = data['close'].shift(2) / data['close'].shift(3)
    
    data['price_accel_t'] = data['price_ratio_1'] - data['price_ratio_2']
    data['price_accel_t1'] = data['price_ratio_2'] - data['price_ratio_3']
    
    # Calculate volume momentum changes
    data['volume_ratio_1'] = data['volume'] / data['volume'].shift(1)
    data['volume_ratio_2'] = data['volume'].shift(1) / data['volume'].shift(2)
    data['volume_ratio_3'] = data['volume'].shift(2) / data['volume'].shift(3)
    
    data['volume_accel_t'] = data['volume_ratio_1'] - data['volume_ratio_2']
    data['volume_accel_t1'] = data['volume_ratio_2'] - data['volume_ratio_3']
    
    # Price-Volume acceleration divergence
    data['pv_divergence'] = data['price_accel_t'] - data['volume_accel_t']
    
    # Volume concentration ratio (t-4 to t)
    data['volume_rolling_sum'] = data['volume'].rolling(window=5, min_periods=3).sum()
    data['volume_concentration'] = data['volume'] / data['volume_rolling_sum']
    
    # Cross-dimensional diffusion coefficient
    data['momentum_diffusion'] = (data['price_accel_t'] - data['price_accel_t1']).abs() + \
                                (data['volume_accel_t'] - data['volume_accel_t1']).abs()
    
    # Microstructure Anchoring Component
    # Intraday anchor calculations
    data['mid_range_anchor'] = (data['high'] + data['low']) / 2
    data['volume_weighted_anchor'] = (data['high'] * data['volume'] + data['low'] * data['volume']) / (2 * data['volume'])
    
    # Anchor deviation metrics
    data['mid_range_deviation'] = data['close'] / data['mid_range_anchor'] - 1
    data['volume_weighted_deviation'] = data['close'] / data['volume_weighted_anchor'] - 1
    
    # Gap Efficiency and Reversal Analysis
    # Opening gap momentum
    data['gap_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    # Handle zero range cases
    data.loc[data['high'] == data['low'], 'gap_efficiency'] = 0
    
    # Gap persistence (sign consistency over 3 days)
    data['gap_sign'] = np.sign(data['gap_efficiency'])
    data['gap_persistence'] = (data['gap_sign'] == data['gap_sign'].shift(1)).astype(int) + \
                             (data['gap_sign'] == data['gap_sign'].shift(2)).astype(int)
    data['gap_persistence'] *= data['gap_efficiency'].abs()
    
    # Intraday reversal efficiency
    data['reversal_strength'] = (data['close'] - data['open']).abs() / (data['high'] - data['low'])
    data.loc[data['high'] == data['low'], 'reversal_strength'] = 0
    
    # Adjust by opening gap direction
    data['gap_reversal_combo'] = data['gap_persistence'] * data['reversal_strength'] * np.sign(data['gap_efficiency'])
    
    # 5-day volatility ratio
    data['volatility_5d'] = data['close'].pct_change().rolling(window=5, min_periods=3).std()
    data['volatility_ratio'] = data['volatility_5d'] / data['volatility_5d'].rolling(window=20, min_periods=10).mean()
    
    # Momentum Persistence Component
    # Calculate acceleration streaks
    data['price_accel_sign'] = np.sign(data['price_accel_t'])
    data['volume_accel_sign'] = np.sign(data['volume_accel_t'])
    
    # Price acceleration streak count
    data['price_streak'] = 0
    for i in range(1, len(data)):
        if data['price_accel_sign'].iloc[i] == data['price_accel_sign'].iloc[i-1]:
            data['price_streak'].iloc[i] = data['price_streak'].iloc[i-1] + 1
        else:
            data['price_streak'].iloc[i] = 1
    
    # Volume acceleration streak count
    data['volume_streak'] = 0
    for i in range(1, len(data)):
        if data['volume_accel_sign'].iloc[i] == data['volume_accel_sign'].iloc[i-1]:
            data['volume_streak'].iloc[i] = data['volume_streak'].iloc[i-1] + 1
        else:
            data['volume_streak'].iloc[i] = 1
    
    # Persistence strength indicators
    data['accel_magnitude_accum'] = data['price_accel_t'].abs().rolling(window=5, min_periods=3).sum()
    data['streak_weighted_persistence'] = (data['price_streak'] * data['price_accel_t'].abs() + 
                                          data['volume_streak'] * data['volume_accel_t'].abs()) / 2
    
    # Acceleration regime detection
    data['accel_regime'] = np.where(data['price_accel_t'] > data['price_accel_t'].rolling(window=10, min_periods=5).mean(), 1, -1)
    data['volume_confirmation'] = np.where(data['volume_accel_t'] * data['price_accel_t'] > 0, 1, -1)
    
    # Regime transition probabilities
    data['regime_change'] = (data['accel_regime'] != data['accel_regime'].shift(1)).astype(int)
    data['persistence_breakdown_likelihood'] = data['regime_change'].rolling(window=10, min_periods=5).mean()
    data['acceleration_stability'] = 1 - data['persistence_breakdown_likelihood']
    
    # Final Factor Integration
    # Cross-component alignment signals
    data['accel_anchor_convergence'] = data['pv_divergence'] * data['mid_range_deviation']
    data['gap_accel_consistency'] = np.sign(data['gap_efficiency']) * np.sign(data['price_accel_t'])
    
    # Persistence-acceleration confirmation
    data['persistence_confirmation'] = data['streak_weighted_persistence'] * data['acceleration_stability']
    
    # Combine all components with dynamic integration
    # Base acceleration diffusion component
    acceleration_component = data['pv_divergence'] * data['volume_concentration'] * data['momentum_diffusion']
    
    # Microstructure anchoring adjustments
    microstructure_component = (data['mid_range_deviation'] + data['volume_weighted_deviation']) / 2 * \
                              data['gap_reversal_combo'] * data['volatility_ratio']
    
    # Persistence multipliers
    persistence_component = data['persistence_confirmation'] * data['acceleration_stability']
    
    # Cross-component confirmation strength
    cross_confirmation = (data['accel_anchor_convergence'].abs() + data['gap_accel_consistency'].abs()) / 2
    
    # Final factor calculation
    factor = (acceleration_component * 0.4 + 
              microstructure_component * 0.35 + 
              persistence_component * 0.25) * cross_confirmation
    
    # Clean up and return
    result = pd.Series(factor, index=data.index, name='pv_acceleration_diffusion_anchor')
    return result
