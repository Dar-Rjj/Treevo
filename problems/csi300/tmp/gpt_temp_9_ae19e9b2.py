import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Basic calculations
    data['prev_close'] = data['close'].shift(1)
    data['range'] = data['high'] - data['low']
    data['range_efficiency'] = (data['close'] - data['open']) / (data['range'] + 0.001)
    data['gap_impact_efficiency'] = np.abs(data['open'] - data['prev_close']) / (data['range'] + 0.001)
    
    # Efficiency calculations
    data['efficiency_3d'] = data['range_efficiency'].rolling(3).mean()
    data['efficiency_5d'] = data['range_efficiency'].rolling(5).mean()
    data['efficiency_10d'] = data['range_efficiency'].rolling(10).mean()
    data['efficiency_20d'] = data['range_efficiency'].rolling(20).mean()
    
    # Efficiency acceleration hierarchy
    data['ultra_short_efficiency_acc'] = (data['efficiency_3d'] - data['efficiency_3d'].shift(1)) - (data['efficiency_5d'] - data['efficiency_5d'].shift(1))
    data['short_term_efficiency_acc'] = (data['efficiency_5d'] - data['efficiency_5d'].shift(1)) - (data['efficiency_10d'] - data['efficiency_10d'].shift(1))
    data['medium_term_efficiency_acc'] = (data['efficiency_10d'] - data['efficiency_10d'].shift(1)) - (data['efficiency_20d'] - data['efficiency_20d'].shift(1))
    
    # Efficiency regime classification
    data['accelerating_efficiency'] = ((data['ultra_short_efficiency_acc'] > 0) & 
                                     (data['short_term_efficiency_acc'] > 0) & 
                                     (data['medium_term_efficiency_acc'] > 0)).astype(int)
    data['decelerating_efficiency'] = ((data['ultra_short_efficiency_acc'] < 0) & 
                                     (data['short_term_efficiency_acc'] > 0) & 
                                     (data['medium_term_efficiency_acc'] > 0)).astype(int)
    data['accelerating_inefficiency'] = ((data['ultra_short_efficiency_acc'] < 0) & 
                                       (data['short_term_efficiency_acc'] < 0) & 
                                       (data['medium_term_efficiency_acc'] < 0)).astype(int)
    data['decelerating_inefficiency'] = ((data['ultra_short_efficiency_acc'] > 0) & 
                                       (data['short_term_efficiency_acc'] < 0) & 
                                       (data['medium_term_efficiency_acc'] < 0)).astype(int)
    
    # Structure compression intensity
    data['range_ma_20d'] = data['range'].rolling(20).mean()
    data['structure_compression_intensity'] = 1 / ((data['range'] / (data['range_ma_20d'] + 0.001)) + 0.001)
    
    # Efficiency-structure synchronization scoring
    efficiency_acc_signs = np.sign(data[['ultra_short_efficiency_acc', 'short_term_efficiency_acc', 'medium_term_efficiency_acc']])
    range_efficiency_change = data['range_efficiency'] - data['range_efficiency'].shift(1)
    range_efficiency_sign = np.sign(range_efficiency_change)
    
    data['direction_sync'] = (efficiency_acc_signs.eq(range_efficiency_sign, axis=0).sum(axis=1) / 3.0)
    
    magnitude_ratios = []
    for i in range(len(data)):
        if i < 1:
            magnitude_ratios.append(0)
            continue
        acc_magnitudes = np.abs([data['ultra_short_efficiency_acc'].iloc[i], 
                               data['short_term_efficiency_acc'].iloc[i], 
                               data['medium_term_efficiency_acc'].iloc[i]])
        range_magnitude = np.abs(range_efficiency_change.iloc[i]) + 0.001
        magnitude_ratios.append(np.mean(acc_magnitudes / range_magnitude))
    data['magnitude_sync'] = magnitude_ratios
    
    # Timing synchronization (5-day correlation)
    timing_corrs = []
    for i in range(len(data)):
        if i < 5:
            timing_corrs.append(0)
            continue
        window = data.iloc[i-4:i+1]
        if len(window) < 5:
            timing_corrs.append(0)
            continue
        try:
            corr = np.corrcoef(window['ultra_short_efficiency_acc'].values, 
                             window['range_efficiency'].values)[0,1]
            timing_corrs.append(corr if not np.isnan(corr) else 0)
        except:
            timing_corrs.append(0)
    data['timing_sync'] = timing_corrs
    
    # Volatility calculations
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['atr_14d'] = data['true_range'].rolling(14).mean()
    data['return_vol_20d'] = data['close'].pct_change().rolling(20).std()
    data['volatility_acceleration'] = data['atr_14d'] - data['atr_14d'].shift(5)
    
    # Volatility regime classification
    data['atr_ma_20d'] = data['atr_14d'].rolling(20).mean()
    data['high_volatility'] = (data['atr_14d'] > data['atr_ma_20d']).astype(int)
    data['low_volatility'] = (data['atr_14d'] < data['atr_ma_20d']).astype(int)
    threshold = 0.1 * data['atr_ma_20d']
    data['transition_volatility'] = (np.abs(data['atr_14d'] - data['atr_ma_20d']) < threshold).astype(int)
    
    # Volume calculations
    data['volume_3d'] = data['volume'].rolling(3).mean()
    data['volume_5d'] = data['volume'].rolling(5).mean()
    data['volume_10d'] = data['volume'].rolling(10).mean()
    
    data['ultra_short_vol_acc'] = (data['volume_3d'] - data['volume_3d'].shift(1)) - (data['volume_5d'] - data['volume_5d'].shift(1))
    data['short_term_vol_acc'] = (data['volume_5d'] - data['volume_5d'].shift(1)) - (data['volume_10d'] - data['volume_10d'].shift(1))
    
    # Volume regime classification
    data['volume_expansion'] = ((data['ultra_short_vol_acc'] > 0) & (data['short_term_vol_acc'] > 0)).astype(int)
    data['volume_contraction'] = ((data['ultra_short_vol_acc'] < 0) & (data['short_term_vol_acc'] < 0)).astype(int)
    data['volume_instability'] = ((data['ultra_short_vol_acc'] * data['short_term_vol_acc']) < 0).astype(int)
    
    # Volume-structure synchronization
    data['volume_range_alignment'] = np.sign(data['ultra_short_vol_acc']) * np.sign(range_efficiency_change)
    
    # Volume confirmation strength
    data['volume_confirmation_strength'] = data['ultra_short_vol_acc'] * range_efficiency_change
    
    # Range efficiency momentum scoring
    positive_efficiency_count = (data[['efficiency_3d', 'efficiency_5d', 'efficiency_10d']] > 0).sum(axis=1)
    data['timeframe_alignment_score'] = positive_efficiency_count / 3.0
    
    # Efficiency persistence
    efficiency_persistence = []
    current_streak = 0
    for i in range(len(data)):
        if i == 0:
            efficiency_persistence.append(0)
            continue
        if data['range_efficiency'].iloc[i] * data['range_efficiency'].iloc[i-1] > 0:
            current_streak += 1
        else:
            current_streak = 1
        efficiency_persistence.append(current_streak)
    data['efficiency_persistence'] = efficiency_persistence
    
    data['range_efficiency_quality'] = (data['short_term_efficiency_acc'] * data['efficiency_persistence']) / (np.abs(data['ultra_short_efficiency_acc']) + 0.001)
    
    # Breakout quality metrics
    data['compression_release_intensity'] = data['structure_compression_intensity'] * data['range_efficiency']
    data['breakout_quality'] = data['compression_release_intensity'] * data['volume_confirmation_strength']
    data['volatility_context'] = data['breakout_quality'] / (data['atr_14d'] + 0.001)
    
    # Multi-timeframe convergence
    data['multi_timeframe_convergence'] = data['direction_sync'] * data['magnitude_sync']
    
    # Volatility-regime multipliers
    data['volatility_regime_multiplier'] = (
        data['high_volatility'] * 0.7 + 
        data['low_volatility'] * 1.3 + 
        data['transition_volatility'] * 1.0
    )
    
    # Efficiency-structure regime adjustments
    data['efficiency_regime_adjustment'] = (
        data['accelerating_efficiency'] * (1 + data['range_efficiency_quality']) +
        data['decelerating_efficiency'] * (1 - data['range_efficiency_quality']) +
        data['accelerating_inefficiency'] * (1 - data['range_efficiency_quality']) +
        data['decelerating_inefficiency'] * (1 + data['range_efficiency_quality'])
    )
    
    # Core synchronization factor
    data['base_sync'] = data['direction_sync'] * data['magnitude_sync'] * data['timing_sync']
    data['volatility_enhanced_sync'] = data['base_sync'] * data['volatility_regime_multiplier']
    data['quality_adjusted_efficiency'] = data['efficiency_regime_adjustment'] * data['range_efficiency_quality']
    
    # Multi-scale structure-efficiency integration
    data['ultra_short_contribution'] = data['ultra_short_efficiency_acc'] * data['structure_compression_intensity']
    data['short_term_contribution'] = data['short_term_efficiency_acc'] * data['range_efficiency_quality']
    data['medium_term_foundation'] = data['medium_term_efficiency_acc'] * data['volume_confirmation_strength']
    
    # Final alpha generation
    data['synchronized_structure_signal'] = data['quality_adjusted_efficiency'] * data['base_sync']
    data['volatility_optimized_timing'] = data['synchronized_structure_signal'] * data['volatility_regime_multiplier']
    data['breakout_confirmed_prediction'] = data['volatility_optimized_timing'] * data['breakout_quality']
    data['final_alpha'] = data['breakout_confirmed_prediction'] * data['volume_confirmation_strength']
    
    # Clean up and return
    result = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
