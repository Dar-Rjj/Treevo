import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price-Volume Asymmetry Ratio with Directional Persistence
    # Short-term asymmetry
    data['price_change_1d'] = data['close'].diff()
    data['short_term_asymmetry'] = (data['close'] - data['close'].shift(5)) / \
                                  data['price_change_1d'].abs().rolling(window=5).sum()
    
    # Medium-term asymmetry
    data['medium_term_asymmetry'] = (data['close'] - data['close'].shift(20)) / \
                                   data['price_change_1d'].abs().rolling(window=20).sum()
    
    # Long-term asymmetry
    data['long_term_asymmetry'] = (data['close'] - data['close'].shift(60)) / \
                                 data['price_change_1d'].abs().rolling(window=60).sum()
    
    # Volume-directional persistence
    data['avg_volume_5d'] = data['volume'].shift(1).rolling(window=5).mean()
    data['volume_trend_asymmetry'] = np.sign(data['price_change_1d']) * \
                                    (data['volume'] / data['avg_volume_5d'])
    
    # Persistence strength
    data['price_volume_sign'] = np.sign(data['price_change_1d'] * data['volume_trend_asymmetry'])
    persistence_count = []
    current_count = 0
    for sign in data['price_volume_sign']:
        if not np.isnan(sign):
            if sign == data['price_volume_sign'].shift(1).iloc[-1] if len(persistence_count) > 0 else False:
                current_count += 1
            else:
                current_count = 1
        else:
            current_count = 0
        persistence_count.append(current_count)
    data['persistence_strength'] = persistence_count
    
    # Asymmetry acceleration
    data['asymmetry_acceleration'] = data['short_term_asymmetry'].diff()
    
    # Regime-Aware Liquidity Momentum
    data['avg_volume_20d'] = data['volume'].shift(1).rolling(window=20).mean()
    data['liquidity_regime'] = np.where(data['volume'] > 1.5 * data['avg_volume_20d'], 'high',
                                      np.where(data['volume'] < 0.7 * data['avg_volume_20d'], 'low', 'normal'))
    
    # Regime-specific liquidity momentum
    data['high_liquidity_momentum'] = (data['close'] - data['close'].shift(5)) * \
                                     (data['volume'] / data['volume'].shift(1).rolling(window=5).mean())
    
    data['low_liquidity_momentum'] = (data['close'] - data['close'].shift(10)) * \
                                    (data['volume'].shift(1).rolling(window=5).mean() / data['volume'])
    
    # Normal momentum as weighted average
    data['normal_momentum'] = 0.6 * data['high_liquidity_momentum'] + 0.4 * data['low_liquidity_momentum']
    
    # Liquidity momentum persistence
    regime_changes = (data['liquidity_regime'] != data['liquidity_regime'].shift(1)).cumsum()
    data['regime_duration'] = regime_changes.groupby(regime_changes).cumcount() + 1
    
    data['liquidity_momentum'] = np.where(data['liquidity_regime'] == 'high', data['high_liquidity_momentum'],
                                        np.where(data['liquidity_regime'] == 'low', data['low_liquidity_momentum'], 
                                                data['normal_momentum']))
    
    data['momentum_intensity'] = np.abs(data['liquidity_momentum']) * data['regime_duration']
    data['momentum_inflection'] = np.sign(data['liquidity_momentum']) != np.sign(data['liquidity_momentum'].shift(1))
    
    # Gap Dynamics and Absorption Analysis
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_gap_fill'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Gap persistence
    gap_direction = np.sign(data['opening_gap'])
    gap_persistence = []
    current_count = 0
    for direction in gap_direction:
        if not np.isnan(direction):
            if direction == gap_direction.shift(1).iloc[-1] if len(gap_persistence) > 0 else False:
                current_count += 1
            else:
                current_count = 1
        else:
            current_count = 0
        gap_persistence.append(current_count)
    data['gap_persistence'] = gap_persistence
    
    # Volume-gap interaction
    data['gap_absorption'] = data['volume'] * np.abs(data['opening_gap'])
    data['gap_rejection'] = (np.sign(data['opening_gap']) != np.sign(data['close'] - data['open'])) & \
                           (data['volume'] > 1.2 * data['avg_volume_5d'])
    data['gap_confirmation'] = (np.sign(data['opening_gap']) == np.sign(data['close'] - data['open'])) & \
                              (data['volume'] > data['volume'].shift(1))
    
    # Multi-day gap patterns
    data['gap_clustering'] = (np.sign(data['opening_gap']) == np.sign(data['opening_gap'].shift(1))).rolling(window=5).sum()
    data['gap_exhaustion'] = (np.abs(data['opening_gap']) < np.abs(data['opening_gap'].shift(1))) & \
                            (data['volume'] > 1.3 * data['avg_volume_5d'])
    data['gap_acceleration'] = (np.abs(data['opening_gap']) > np.abs(data['opening_gap'].shift(1))) & \
                              data['gap_confirmation']
    
    # Price-Range Efficiency with Volatility Compression
    data['range_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['range_consistency_5d'] = data['range_efficiency'].rolling(window=5).std()
    data['range_stability_20d'] = data['range_efficiency'].rolling(window=20).mean()
    
    # Volatility compression dynamics
    data['avg_range_5d'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['avg_range_20d'] = (data['high'] - data['low']).rolling(window=20).mean()
    data['range_compression'] = data['avg_range_5d'] / data['avg_range_20d']
    
    # Volatility clustering (simplified)
    high_vol_threshold = data['avg_range_20d'].quantile(0.7)
    high_vol_periods = (data['avg_range_5d'] > high_vol_threshold)
    data['days_since_high_vol'] = (~high_vol_periods).cumsum() - (~high_vol_periods).cumsum().where(high_vol_periods).ffill().fillna(0)
    
    data['compression_intensity'] = data['range_compression'] * data['days_since_high_vol']
    
    # Breakout quality metrics
    data['breakout_strength'] = (data['close'] - data['open']) * data['range_compression']
    data['breakout_validation'] = data['volume'] > 1.2 * data['avg_volume_5d']
    data['false_breakout'] = ((data['high'] - data['low']) > data['avg_range_5d']) & \
                            (np.abs(data['close'] - data['open']) < 0.3 * (data['high'] - data['low']))
    
    # Cross-Timeframe Signal Asymmetry
    data['short_term_alignment'] = data['short_term_asymmetry'] * data['high_liquidity_momentum']
    data['medium_term_alignment'] = data['medium_term_asymmetry'] * data['liquidity_momentum']
    
    data['gap_direction_volume'] = data['volume'] * np.sign(data['opening_gap'])
    data['momentum_gap_alignment'] = data['liquidity_momentum'] * data['intraday_gap_fill']
    
    # Regime-context signal weighting
    data['high_liquidity_weight'] = data['volume'] / data['avg_volume_20d']
    data['low_liquidity_weight'] = data['avg_volume_5d'] / data['volume']
    data['compression_weight'] = data['range_compression']
    
    # Adaptive Alpha Synthesis with Dynamic Weighting
    # Core asymmetry momentum signal
    data['core_asymmetry_momentum'] = data['liquidity_momentum'] * data['short_term_asymmetry'] * data['persistence_strength']
    
    # Gap dynamics enhancement
    gap_enhancement = np.where(data['gap_confirmation'], 1.2,
                             np.where(data['gap_rejection'], 0.8, 1.0))
    data['gap_enhanced_signal'] = data['core_asymmetry_momentum'] * gap_enhancement * data['gap_absorption']
    
    # Range efficiency timing
    range_timing = np.where(data['range_compression'] < 0.8, 1.3, 1.0)
    breakout_boost = np.where(data['breakout_validation'] & ~data['false_breakout'], 1.2, 1.0)
    data['range_timed_signal'] = data['gap_enhanced_signal'] * range_timing * breakout_boost * data['range_efficiency']
    
    # Cross-timeframe asymmetry scoring
    timeframe_alignment = (np.sign(data['short_term_asymmetry']) == np.sign(data['medium_term_asymmetry'])).astype(int) + \
                         (np.sign(data['short_term_asymmetry']) == np.sign(data['long_term_asymmetry'])).astype(int)
    
    gap_confirmation_score = data['gap_clustering'] * data['gap_acceleration'].astype(int)
    
    data['asymmetry_score'] = timeframe_alignment + gap_confirmation_score
    
    # Final adaptive alpha with dynamic weighting
    regime_weights = np.where(data['liquidity_regime'] == 'high', data['high_liquidity_weight'],
                            np.where(data['liquidity_regime'] == 'low', data['low_liquidity_weight'], 1.0))
    
    final_alpha = (data['range_timed_signal'] * regime_weights * 
                  data['asymmetry_score'] * data['compression_weight'])
    
    # Clean up and return
    alpha_series = final_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_series
