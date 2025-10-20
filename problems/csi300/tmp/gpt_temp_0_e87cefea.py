import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Acceleration-Efficiency Framework
    # Multi-Timeframe Acceleration
    data['short_term_accel'] = (data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))
    data['medium_momentum_div'] = ((data['close']/data['close'].shift(5) - 1) - 
                                  (data['close']/data['close'].shift(10) - 1))
    data['accel_persistence'] = data['close'].rolling(window=5).apply(
        lambda x: np.sum(x > x.shift(1)) / 5 if len(x) == 5 else np.nan, raw=False)
    
    # Efficiency Dynamics
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # VWAP efficiency calculation
    data['close_vol_3'] = data['close'] * data['volume']
    data['vwap_3'] = (data['close_vol_3'].rolling(window=3).sum() / 
                      data['volume'].rolling(window=3).sum()).replace(0, np.nan)
    data['vwap_efficiency'] = (data['close'] - data['vwap_3']) / data['close']
    
    data['hl_range'] = data['high'] - data['low']
    data['ma_hl_5'] = data['hl_range'].rolling(window=5).mean().shift(1)
    data['compression_efficiency'] = (data['hl_range'] / data['ma_hl_5'].replace(0, np.nan)) * data['range_efficiency']
    
    # Acceleration-Efficiency Alignment
    data['eff_weighted_accel'] = data['short_term_accel'] * data['range_efficiency']
    data['volume_confirmed_div'] = data['medium_momentum_div'] * data['vwap_efficiency']
    data['compression_enhanced_pers'] = data['accel_persistence'] * data['compression_efficiency']
    
    # Volume-Pressure Cumulation
    # Volume Dynamics
    data['volume_accel'] = ((data['volume']/data['volume'].shift(5) - 
                            data['volume'].shift(3)/data['volume'].shift(8)) / 3)
    
    data['volume_pressure'] = (data['volume'] * (2*data['close'] - data['high'] - data['low']) / 
                              (data['high'] - data['low']).replace(0, np.nan))
    
    data['ma_volume_5'] = data['volume'].rolling(window=5).mean().shift(1)
    data['volume_reversal'] = (data['volume'] > 1.2 * data['ma_volume_5']).astype(float)
    
    # Pressure Cumulation
    data['net_pressure'] = (2*data['close'] - data['high'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['pressure_momentum'] = data['net_pressure'] - data['net_pressure'].shift(3)
    data['cumulative_pressure'] = ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)).rolling(window=3).sum()
    
    # Volume-Pressure Synchronization
    data['volume_weighted_pressure'] = data['volume_pressure'] * data['pressure_momentum']
    data['accel_pressure_alignment'] = data['short_term_accel'] * data['cumulative_pressure']
    data['volume_confirmed_eff'] = data['vwap_efficiency'] * data['volume_reversal']
    
    # Volatility-Regime Classification
    # Volatility Metrics
    data['vol_ratio'] = data['close'].rolling(window=5).std() / data['close'].rolling(window=10).std()
    data['range_vol'] = (data['hl_range'] / data['close']) / (data['hl_range'] / data['close']).rolling(window=10).mean().shift(1)
    data['vol_regime'] = data['close'].rolling(window=5).std() / data['close'].rolling(window=20).std()
    
    # Regime Determination
    ma_range_vol_20 = (data['hl_range'] / data['close']).rolling(window=20).mean().shift(1)
    data['high_vol_regime'] = ((data['vol_ratio'] > 1) & (data['range_vol'] > 1)).astype(float)
    data['low_vol_regime'] = ((data['vol_ratio'] <= 1) & (data['range_vol'] <= 1)).astype(float)
    data['extreme_vol_regime'] = (data['range_vol'] > 2 * ma_range_vol_20).astype(float)
    
    # Divergence Detection & Enhancement
    # Price Reversal Components
    def count_recent_reversals(window):
        if len(window) < 5:
            return np.nan
        count = 0
        for i in range(1, 5):
            if (window.iloc[i] > window.iloc[i-1]) and (window.iloc[i] < window.iloc[i-1]):
                count += 1
        return count / 4
    
    data['recent_reversals'] = data['high'].rolling(window=5).apply(count_recent_reversals, raw=False)
    data['gap_recovery'] = (data['open'] - data['close'].shift(1)) / data['hl_range'].replace(0, np.nan)
    
    def count_failed_breakouts(window):
        if len(window) < 3:
            return np.nan
        count = 0
        for i in range(1, 3):
            if (window.iloc[i] > window.iloc[i-1]) and (window.iloc[i] < window.iloc[i-1]):
                count += 1
        return count / 2
    
    data['failed_breakouts'] = data['high'].rolling(window=3).apply(count_failed_breakouts, raw=False)
    
    # Volume Divergence Signals
    data['volume_accel_div'] = data['volume_accel'] * data['volume_reversal']
    data['pressure_volume_align'] = data['volume_pressure'] * data['pressure_momentum']
    data['efficiency_volume_conf'] = data['vwap_efficiency'] * data['volume_accel']
    
    # Enhanced Divergence Framework
    data['reversal_weighted_accel'] = data['short_term_accel'] * data['recent_reversals']
    data['gap_recovery_div'] = data['medium_momentum_div'] * data['gap_recovery']
    data['volume_efficiency_div'] = data['volume_accel'] * data['compression_efficiency']
    
    # Multi-Dimensional Signal Integration
    # Acceleration-Efficiency-Pressure Convergence
    data['primary_convergence'] = data['eff_weighted_accel'] * data['volume_weighted_pressure']
    data['secondary_alignment'] = data['volume_confirmed_div'] * data['accel_pressure_alignment']
    data['tertiary_enhancement'] = data['compression_enhanced_pers'] * data['volume_efficiency_div']
    
    # Regime-Adaptive Alpha Synthesis
    # High Volatility Regime
    high_vol_primary = data['volume_accel_div'] * data['pressure_volume_align']
    high_vol_secondary = data['reversal_weighted_accel'] * data['gap_recovery_div']
    data['high_vol_factor'] = high_vol_primary * 0.6 + high_vol_secondary * 0.4
    
    # Low Volatility Regime
    low_vol_primary = data['eff_weighted_accel'] * data['volume_confirmed_eff']
    low_vol_secondary = data['compression_enhanced_pers'] * data['vwap_efficiency']
    data['low_vol_factor'] = low_vol_primary * 0.5 + low_vol_secondary * 0.5
    
    # Extreme Volatility Filtering
    extreme_primary = data['primary_convergence'] * 0.3
    extreme_secondary = data['secondary_alignment'] * 0.7
    data['extreme_vol_factor'] = (extreme_primary + extreme_secondary) * 0.5
    
    # Unified Alpha Framework with smooth transitions
    data['final_alpha'] = (
        data['high_vol_regime'] * data['high_vol_factor'] +
        data['low_vol_regime'] * data['low_vol_factor'] +
        data['extreme_vol_regime'] * data['extreme_vol_factor']
    )
    
    # Apply volume-confirmed divergence as confidence weighting
    confidence_weight = 1 + data['volume_confirmed_div'].abs()
    data['final_alpha'] = data['final_alpha'] * confidence_weight
    
    # Range-efficiency context for signal calibration
    range_efficiency_weight = 1 + data['range_efficiency'].abs()
    data['final_alpha'] = data['final_alpha'] * range_efficiency_weight
    
    # Acceleration-persistence weighted directional prediction
    persistence_weight = 1 + data['accel_persistence']
    data['final_alpha'] = data['final_alpha'] * persistence_weight
    
    return data['final_alpha']
