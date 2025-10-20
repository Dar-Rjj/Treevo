import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multi-Scale Efficiency Dynamics
    # Efficiency calculation: (close - open) / (high - low + 0.001)
    data['efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    
    # Efficiency acceleration hierarchy
    data['efficiency_3d'] = data['efficiency'].rolling(window=3).mean()
    data['efficiency_5d'] = data['efficiency'].rolling(window=5).mean()
    data['efficiency_10d'] = data['efficiency'].rolling(window=10).mean()
    data['efficiency_20d'] = data['efficiency'].rolling(window=20).mean()
    
    data['ultra_short_acc'] = data['efficiency_3d'] - data['efficiency_5d']
    data['short_term_acc'] = data['efficiency_5d'] - data['efficiency_10d']
    data['medium_term_acc'] = data['efficiency_10d'] - data['efficiency_20d']
    
    # Efficiency regime classification
    conditions = [
        (data['ultra_short_acc'] > 0) & (data['short_term_acc'] > 0) & (data['medium_term_acc'] > 0),
        (data['ultra_short_acc'] < 0) & (data['short_term_acc'] < 0) & (data['medium_term_acc'] < 0)
    ]
    choices = [2, -1]  # Accelerating: 2, Decelerating: -1, Mixed: 0
    data['efficiency_regime'] = np.select(conditions, choices, default=0)
    
    # Structure Analysis Framework
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['gap_impact'] = abs(data['open'] - data['prev_close']) / (data['high'] - data['low'] + 0.001)
    
    # Structure compression
    data['daily_range'] = data['high'] - data['low']
    data['range_20d_ma'] = data['daily_range'].rolling(window=20).mean()
    data['structure_compression'] = 1 / (data['daily_range'] / data['range_20d_ma'] + 0.001)
    
    # Volatility Regime Detection
    data['atr_14d'] = data['true_range'].rolling(window=14).mean()
    data['atr_20d_ma'] = data['atr_14d'].rolling(window=20).mean()
    data['volatility_regime'] = np.where(data['atr_14d'] > data['atr_20d_ma'], 1, 0)  # 1: high, 0: low
    
    # Volatility acceleration
    data['atr_5d_change'] = data['atr_14d'].diff(5)
    
    # Volume Confirmation
    data['volume_3d_change'] = data['volume'].pct_change(3)
    data['volume_5d_change'] = data['volume'].pct_change(5)
    data['volume_acceleration'] = data['volume_3d_change'] - data['volume_5d_change']
    
    data['range_efficiency_change'] = data['range_efficiency'].diff()
    data['volume_structure_alignment'] = np.sign(data['volume_acceleration']) * np.sign(data['range_efficiency_change'])
    
    # Breakout Quality Assessment
    # Efficiency persistence
    data['regime_persistence'] = 0
    current_regime = data['efficiency_regime'].iloc[0]
    count = 0
    for i in range(len(data)):
        if data['efficiency_regime'].iloc[i] == current_regime:
            count += 1
        else:
            current_regime = data['efficiency_regime'].iloc[i]
            count = 1
        data.loc[data.index[i], 'regime_persistence'] = count
    
    # Compression-release
    data['compression_release'] = data['structure_compression'] * data['range_efficiency']
    
    # Volume confirmation
    data['volume_confirmation'] = data['volume_acceleration'] * data['range_efficiency_change']
    
    # Adaptive Signal Construction
    # Volatility multipliers
    data['volatility_multiplier'] = np.where(data['volatility_regime'] == 1, 0.7, 1.3)
    
    # Regime adjustments
    data['efficiency_adjustment'] = np.where(data['efficiency_regime'] == 2, 1 + data['range_efficiency'], 1)
    data['structure_breakout_adjustment'] = 1 + data['compression_release']
    data['volume_confirmation_adjustment'] = 1 + data['volume_confirmation']
    
    # Composite Alpha Generation
    # Base synchronization
    data['base_sync'] = data['efficiency_regime'] * data['range_efficiency'] * data['structure_compression']
    
    # Volatility adaptation
    data['volatility_adapted'] = data['base_sync'] * data['volatility_multiplier']
    
    # Quality enhancement
    data['breakout_quality'] = data['regime_persistence'] * data['compression_release'] * data['volume_confirmation']
    data['quality_enhanced'] = data['volatility_adapted'] * data['breakout_quality']
    
    # Final alpha
    data['alpha'] = data['quality_enhanced'] * data['volume_structure_alignment']
    
    # Clean up intermediate columns
    result = data['alpha'].copy()
    
    return result
