import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multi-Timeframe Gap Fractality
    data['gap_fractal_3d'] = abs(data['close'] - data['close'].shift(2)) / (
        data['true_range'].rolling(window=3, min_periods=3).sum()
    )
    data['gap_fractal_5d'] = abs(data['close'] - data['close'].shift(4)) / (
        data['true_range'].rolling(window=5, min_periods=5).sum()
    )
    data['fractal_momentum_gap'] = data['gap_fractal_5d'] - data['gap_fractal_3d']
    
    # Fractal Efficiency Dynamics
    data['gap_fractal_8d'] = abs(data['close'] - data['close'].shift(7)) / (
        data['true_range'].rolling(window=8, min_periods=8).sum()
    )
    data['efficiency_spread'] = data['gap_fractal_3d'] - data['gap_fractal_8d']
    data['fractal_convergence'] = data['fractal_momentum_gap'] * data['efficiency_spread']
    
    # Fractal Persistence
    data['fractal_persistence'] = data['fractal_momentum_gap'].rolling(window=5).corr(data['efficiency_spread'])
    
    # Anchored Volume Signature Framework
    data['daily_anchor'] = (data['high'] * data['volume'] + data['low'] * data['volume']) / (2 * data['volume'])
    data['rolling_anchor_3d'] = data['daily_anchor'].rolling(window=3, min_periods=3).mean()
    data['anchor_divergence'] = (data['close'] - data['daily_anchor']) / data['close']
    
    # Volume Pressure Dynamics
    data['up_day'] = data['close'] > data['open']
    data['down_day'] = data['close'] < data['open']
    
    up_volume_5d = []
    down_volume_5d = []
    total_volume_5d = []
    
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            up_vol = window_data.loc[window_data['up_day'], 'volume'].sum()
            down_vol = window_data.loc[window_data['down_day'], 'volume'].sum()
            total_vol = window_data['volume'].sum()
        else:
            up_vol = down_vol = total_vol = np.nan
        
        up_volume_5d.append(up_vol)
        down_volume_5d.append(down_vol)
        total_volume_5d.append(total_vol)
    
    data['up_volume_5d'] = up_volume_5d
    data['down_volume_5d'] = down_volume_5d
    data['total_volume_5d'] = total_volume_5d
    
    data['up_day_volume_intensity'] = data['up_volume_5d'] / data['total_volume_5d']
    data['down_day_volume_intensity'] = data['down_volume_5d'] / data['total_volume_5d']
    data['volume_pressure_asymmetry'] = data['up_day_volume_intensity'] - data['down_day_volume_intensity']
    
    data['volume_quality_score'] = (data['volume'] / (data['high'] - data['low'])) * (
        data['volume'] / data['volume'].shift(5) - 1
    )
    
    # Gap Persistence with Volume Confirmation
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_realization'] = (data['close'] - data['open']) / data['open']
    data['gap_persistence'] = np.sign(data['overnight_gap']) * np.sign(data['intraday_realization'])
    
    data['gap_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['gap_strength'] = abs(data['overnight_gap']) / (1 + abs(data['intraday_realization']))
    data['volume_confirmed_gap_quality'] = data['gap_efficiency'] * data['gap_strength'] * data['volume_pressure_asymmetry']
    
    # Fractal-Anchor Divergence Detection
    data['short_term_anchor_divergence'] = data['gap_fractal_3d'] * data['anchor_divergence']
    data['medium_term_anchor_divergence'] = data['gap_fractal_5d'] * data['anchor_divergence']
    data['fractal_anchor_convergence'] = data['short_term_anchor_divergence'] - data['medium_term_anchor_divergence']
    
    # Volume-Price Fractal Anomaly
    data['fractal_up_volume_down'] = (data['fractal_momentum_gap'] > 0) & (data['volume'] < data['volume'].shift(1))
    data['fractal_down_volume_up'] = (data['fractal_momentum_gap'] < 0) & (data['volume'] > data['volume'].shift(1))
    
    fractal_up_count = []
    fractal_down_count = []
    
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            up_count = window_data['fractal_up_volume_down'].sum()
            down_count = window_data['fractal_down_volume_up'].sum()
        else:
            up_count = down_count = np.nan
        
        fractal_up_count.append(up_count)
        fractal_down_count.append(down_count)
    
    data['fractal_up_count_5d'] = fractal_up_count
    data['fractal_down_count_5d'] = fractal_down_count
    data['fractal_anomaly_strength'] = data['fractal_up_count_5d'] - data['fractal_down_count_5d']
    
    # Volatility-Fractal Regime Framework
    data['atr_3'] = data['true_range'].rolling(window=3, min_periods=3).mean()
    data['atr_8'] = data['true_range'].rolling(window=8, min_periods=8).mean()
    data['short_term_volatility'] = data['atr_3'] / data['close']
    data['medium_term_volatility'] = data['atr_8'] / data['close']
    data['volatility_regime_score'] = data['short_term_volatility'] / data['medium_term_volatility']
    
    # Microstructure Quality Filtering
    data['upper_shadow_rejection'] = (data['high'] - data[['open', 'close']].max(axis=1)) / (data['high'] - data['low'])
    data['lower_shadow_rejection'] = (data[['open', 'close']].min(axis=1) - data['low']) / (data['high'] - data['low'])
    data['net_rejection_pressure'] = data['upper_shadow_rejection'] - data['lower_shadow_rejection']
    
    data['volume_surge'] = data['volume'] / data['volume'].rolling(window=5, min_periods=5).mean().shift(1)
    data['volume_distribution'] = ((data['high'] - data['close']) / (data['high'] - data['low'])) - (
        (data['close'] - data['low']) / (data['high'] - data['low'])
    )
    data['volume_quality_filter'] = data['volume_surge'] * data['volume_distribution']
    
    # Core Signal Integration
    data['raw_fractal_signal'] = data['fractal_convergence'] * data['gap_persistence']
    data['volume_enhanced_fractal'] = data['raw_fractal_signal'] * data['volume_pressure_asymmetry'] * data['volume_quality_score']
    data['anchor_confirmed_fractal'] = data['volume_enhanced_fractal'] * data['anchor_divergence']
    
    data['anomaly_confirmation'] = data['anchor_confirmed_fractal'] * data['fractal_anomaly_strength']
    data['fractal_persistence_confirmation'] = data['anomaly_confirmation'] * data['fractal_persistence']
    data['gap_quality_confirmation'] = data['fractal_persistence_confirmation'] * data['volume_confirmed_gap_quality']
    
    data['fractal_anchor_enhancement'] = data['gap_quality_confirmation'] * data['fractal_anchor_convergence']
    data['volume_distribution_enhancement'] = data['fractal_anchor_enhancement'] * data['volume_quality_filter']
    
    # Adaptive Final Integration
    data['rejection_filter'] = 1 / (1 + abs(data['net_rejection_pressure']))
    data['combined_quality_filter'] = data['rejection_filter'] * data['volume_quality_filter']
    
    # Regime-Adaptive Scaling
    high_fractal_vol = (data['volatility_regime_score'] > 1.5) & (data['fractal_momentum_gap'] > 0.1)
    low_fractal_vol = (data['volatility_regime_score'] < 0.7) & (data['fractal_momentum_gap'] < -0.1)
    
    data['regime_scale_factor'] = 0.5  # Normal Fractal Volatility
    data.loc[high_fractal_vol, 'regime_scale_factor'] = 0.3
    data.loc[low_fractal_vol, 'regime_scale_factor'] = 0.7
    
    # Final Alpha Output
    data['filtered_signal'] = data['volume_distribution_enhancement'] * data['combined_quality_filter']
    data['final_output'] = data['filtered_signal'] * data['regime_scale_factor']
    
    return data['final_output']
