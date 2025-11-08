import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility State Classification
    # Price Volatility Regimes
    data['range_efficiency'] = ((data['high'] - data['low']) / data['close']) * \
                              ((data['close'] - data['open']) / (data['high'] - data['low']))
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volatility Persistence
    data['vol_persistence'] = 0
    for i in range(4, len(data)):
        window = data['range_efficiency'].iloc[i-4:i+1]
        if len(window) == 5:
            signs = np.sign(window.diff().dropna())
            data.iloc[i, data.columns.get_loc('vol_persistence')] = (signs == 0).sum()
    
    # Multi-day Volatility Dynamics
    data['vol_momentum'] = data['range_efficiency'] - data['range_efficiency'].shift(4)
    
    # Volume Volatility Framework
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_intensity'] = data['volume'] / data['volume_ma_10']
    data['volume_volatility'] = np.abs(data['volume_intensity'] - 1) * data['volume_intensity']
    
    data['volume_vol_change'] = data['volume_volatility'] - data['volume_volatility'].shift(5)
    data['volume_vol_accel'] = data['volume_vol_change'] - data['volume_vol_change'].shift(1)
    
    # 2. Multi-Frequency Momentum Spectrum
    data['hf_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['mf_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['lf_momentum'] = data['close'] / data['close'].shift(10) - 1
    
    data['momentum_slope'] = (data['hf_momentum'] + data['mf_momentum']) / 2 - data['lf_momentum']
    data['momentum_curvature'] = (data['hf_momentum'] - data['mf_momentum']) - (data['mf_momentum'] - data['lf_momentum'])
    
    # Volatility-Momentum Coupling
    data['vol_direction'] = np.sign(data['vol_momentum'])
    data['mom_direction'] = np.sign(data['mf_momentum'])
    data['phase_alignment'] = data['vol_direction'] * data['mom_direction']
    
    data['phase_stability'] = 0
    for i in range(4, len(data)):
        window = data['phase_alignment'].iloc[i-4:i+1]
        data.iloc[i, data.columns.get_loc('phase_stability')] = (window > 0).sum()
    
    # 3. Microstructural Volatility Patterns
    data['pv_vol_corr'] = np.sign(data['vol_momentum']) * np.sign(data['volume_vol_change'])
    data['sync_intensity'] = np.abs(data['vol_momentum']) * np.abs(data['volume_vol_change'])
    
    data['sustained_sync'] = 0
    for i in range(6, len(data)):
        window = data['pv_vol_corr'].iloc[i-6:i+1]
        data.iloc[i, data.columns.get_loc('sustained_sync')] = (window > 0).sum()
    
    data['sync_volatility'] = data['sync_intensity'].rolling(window=7, min_periods=1).var()
    
    # Volatility Efficiency
    data['vol_compression_ratio'] = data['range_efficiency'] / ((data['high'] - data['low']) / data['close'])
    data['vol_compression_ratio'] = data['vol_compression_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['efficiency_consistency'] = 0
    for i in range(4, len(data)):
        window = data['vol_compression_ratio'].iloc[i-4:i+1]
        data.iloc[i, data.columns.get_loc('efficiency_consistency')] = (window > 0.6).sum()
    
    data['vol_vol_stability'] = data['volume_volatility'] / data['volume_volatility'].shift(5)
    data['vol_vol_stability'] = data['vol_vol_stability'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    data['vol_vol_compression'] = data['volume_vol_change'] / data['volume_vol_accel']
    data['vol_vol_compression'] = data['vol_vol_compression'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 4. Regime Adaptive Signal Construction
    # Core Volatility Regime Component
    data['vol_state_score'] = 0.0
    
    # Volatility State Scoring
    high_vol_exp_mask = (data['vol_momentum'] > 0) & (data['phase_alignment'] > 0)
    low_vol_comp_mask = (data['vol_momentum'] < 0) & (data['phase_alignment'] > 0)
    vol_divergence_mask = (data['phase_alignment'] < 0)
    stable_vol_mask = (np.abs(data['vol_momentum']) < data['vol_momentum'].rolling(window=20).std() * 0.5)
    
    data.loc[high_vol_exp_mask, 'vol_state_score'] = 2.2
    data.loc[low_vol_comp_mask, 'vol_state_score'] = 1.6
    data.loc[vol_divergence_mask, 'vol_state_score'] = -1.2
    data.loc[stable_vol_mask & ~high_vol_exp_mask & ~low_vol_comp_mask & ~vol_divergence_mask, 'vol_state_score'] = 0.9
    
    # Momentum Spectrum Integration
    data['core_factor'] = (data['vol_state_score'] + 
                          data['momentum_slope'] * data['vol_momentum'] + 
                          data['momentum_curvature'] * data['volume_vol_change'])
    
    # Microstructure Enhancement
    strong_sync_mask = data['sustained_sync'] > 4
    high_efficiency_mask = data['efficiency_consistency'] > 3
    stable_vol_vol_mask = data['vol_vol_stability'] > 0.85
    
    data.loc[strong_sync_mask, 'core_factor'] *= 1.5
    data.loc[high_efficiency_mask, 'core_factor'] *= 1.4
    data.loc[stable_vol_vol_mask, 'core_factor'] *= 1.3
    
    # Quality Validators
    data['vol_efficiency_filter'] = data['vol_compression_ratio'] * data['efficiency_consistency']
    data['volume_quality_filter'] = data['vol_vol_stability'] * data['vol_vol_compression']
    
    # Frequency Scale Adaptation
    data['final_factor'] = (data['core_factor'] * data['hf_momentum'] * 1.8 + 
                           data['core_factor'] * data['momentum_slope'] * data['vol_momentum'])
    
    # Apply synchronization amplifiers to final factor
    data.loc[strong_sync_mask, 'final_factor'] *= 1.5
    data.loc[high_efficiency_mask, 'final_factor'] *= 1.4
    data.loc[stable_vol_vol_mask, 'final_factor'] *= 1.3
    
    # Apply quality filters
    data['final_factor'] = data['final_factor'] * data['vol_efficiency_filter'] * data['volume_quality_filter']
    
    # Clean up and return
    result = data['final_factor'].fillna(0)
    return result
