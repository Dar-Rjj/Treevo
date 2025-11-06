import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate rolling statistics
    data['vol_5d'] = data['high'].rolling(5).max() - data['low'].rolling(5).min()
    data['close_std_2d'] = data['close'].rolling(2).std()
    data['close_std_9d'] = data['close'].rolling(9).std()
    data['volume_5d'] = data['volume'].rolling(5).mean()
    data['volume_4d'] = data['volume'].rolling(4).mean()
    data['volume_3d'] = data['volume'].rolling(3).mean()
    
    # Regime Transition Detection
    data['regime_momentum'] = (data['close'] - data['close'].shift(5)) / data['vol_5d']
    data['volatility_transition_intensity'] = data['close_std_2d'] / data['close_std_9d']
    data['volume_regime_shift'] = (data['volume'] / data['volume_5d']) - (data['volume'].shift(1) / data['volume_4d'].shift(1))
    data['regime_interaction'] = ((data['high'] - data['low']) / data['close'].shift(1)) * (data['volume'] / data['volume_5d'])
    
    # Flow-Volatility Integration
    data['gap_flow_momentum'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * (data['amount'] / (data['high'] - data['low']))
    data['volatility_efficiency'] = ((data['close'] - data['open']) / (data['high'] - data['low'])) * data['volume']
    data['volume_weighted_pressure'] = (((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])) * data['volume']
    data['flow_volatility_coupling'] = data['gap_flow_momentum'] * data['volatility_efficiency'] * data['volume_weighted_pressure']
    
    # Microstructure Quality Assessment
    data['price_formation_quality'] = np.abs(data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    
    # Volume consistency (3-day vs 5-day correlation)
    def rolling_corr_3_5(x):
        if len(x) < 5:
            return np.nan
        vol_3d = x.rolling(3).mean().dropna()
        vol_5d = x.rolling(5).mean().dropna()
        if len(vol_3d) < 3 or len(vol_5d) < 3:
            return np.nan
        common_idx = vol_3d.index.intersection(vol_5d.index)
        if len(common_idx) < 3:
            return np.nan
        return vol_3d.loc[common_idx].corr(vol_5d.loc[common_idx])
    
    data['volume_consistency'] = data['volume'].rolling(10, min_periods=5).apply(rolling_corr_3_5, raw=False)
    
    # Volatility persistence
    def count_same_direction(x):
        if len(x) < 5:
            return np.nan
        changes = x.diff().dropna()
        if len(changes) < 4:
            return np.nan
        same_dir_count = sum((changes.iloc[1:] * changes.iloc[:-1]) > 0)
        return same_dir_count
    
    data['volatility_persistence'] = (data['high'] - data['low']).rolling(5).apply(count_same_direction, raw=False)
    data['quality_adjusted_flow'] = data['flow_volatility_coupling'] * (1 - np.abs(data['price_formation_quality'])) * data['volume_consistency']
    
    # Regime-Adaptive Signal Construction
    data['high_volatility_regime_factor'] = data['quality_adjusted_flow'] * data['volatility_transition_intensity'] * data['regime_interaction']
    data['low_volatility_regime_factor'] = data['gap_flow_momentum'] * data['volume_weighted_pressure'] * (1 / data['volatility_transition_intensity'])
    data['volume_regime_confirmation'] = data['volatility_efficiency'] * data['volume_regime_shift'] * data['volatility_persistence']
    data['regime_adaptive_core'] = (data['high_volatility_regime_factor'] + data['low_volatility_regime_factor']) * data['volume_regime_confirmation']
    
    # Flow-Volatility Momentum Integration
    data['volatility_volume_momentum'] = ((data['high'] - data['low']) * data['volume'] / data['amount']) * data['regime_momentum']
    
    # Flow-pressure efficiency
    data['flow_pressure_efficiency'] = ((data['amount'] - data['amount'].shift(1)) / (data['amount'].shift(1) + data['amount'].shift(2))) * data['volume_weighted_pressure']
    
    data['composite_momentum'] = data['volatility_volume_momentum'] * data['flow_pressure_efficiency'] * data['quality_adjusted_flow']
    data['momentum_confirmed_signal'] = data['composite_momentum'] * data['volatility_persistence'] * data['volume_consistency']
    
    # Final Alpha Synthesis
    data['regime_adaptive_factor'] = data['regime_adaptive_core'] * data['momentum_confirmed_signal']
    data['quality_enhancement'] = data['regime_adaptive_factor'] * (1 - np.abs(data['price_formation_quality'])) * data['volume_consistency']
    data['final_alpha'] = data['quality_enhancement'] * data['volatility_efficiency'] * data['flow_volatility_coupling']
    
    return data['final_alpha']
