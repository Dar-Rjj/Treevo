import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Liquidity Regime Analysis
    # Asymmetric Volatility Components
    data['upward_vol'] = (data['high'] - data['open']) / data['open']
    data['downward_vol'] = (data['open'] - data['low']) / data['open']
    data['vol_asymmetry'] = data['upward_vol'] / (data['downward_vol'] + 1e-8)
    
    # Intraday Range Efficiency
    data['first_half_range'] = (data['high'] - data['open']) + (data['open'] - data['low'])
    data['second_half_range'] = (data['high'] - data['close']) + (data['close'] - data['low'])
    data['range_shift_ratio'] = data['second_half_range'] / (data['first_half_range'] + 1e-8)
    
    # Volatility Persistence Patterns
    data['daily_range'] = data['high'] - data['low']
    data['vol_regime_indicator'] = data['daily_range'] / (data['daily_range'].shift(1).rolling(window=5).mean() + 1e-8)
    data['vol_momentum'] = data['daily_range'] / (data['daily_range'].shift(1) + 1e-8)
    data['vol_clustering'] = data['close'].rolling(window=5).std() / (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min() + 1e-8)
    
    # Liquidity Flow Dynamics
    # Volume Distribution Analysis
    data['volume_timing_bias'] = (data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + 1e-8)) - (data['volume'] / (data['volume'].shift(1) + 1e-8))
    data['volume_concentration'] = (data['volume'] + data['volume'].shift(1)) / (data['volume'].shift(2) + data['volume'].shift(3) + data['volume'].shift(4) + 1e-8)
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(3)) / (data['volume'].shift(3) + 1e-8)
    
    # Volume Efficiency Metrics
    data['volume_per_unit_move'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['volume_efficiency_div'] = data['volume_per_unit_move'] - data['volume_per_unit_move'].shift(1)
    data['volume_vol_ratio'] = data['volume'] / (data['daily_range'] + 1e-8)
    
    # Liquidity Imbalance Signals
    data['vw_price_pressure'] = (data['close'] - data['open']) * data['volume']
    data['liquidity_momentum'] = data['volume'] / (data['volume'].shift(3) + 1e-8)
    data['volume_regime_persistence'] = data['volume_per_unit_move'] / (data['volume_per_unit_move'].shift(1) + 1e-8)
    
    # Multi-Timeframe Convergence Detection
    # Short-Term Alignment (1-3 days)
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    data['directional_alignment'] = np.sign(data['price_change']) * np.sign(data['volume_change'])
    data['alignment_strength'] = abs(data['price_change']) * abs(data['volume_change'])
    
    # Calculate alignment persistence
    alignment_persistence = []
    for i in range(len(data)):
        if i < 2:
            alignment_persistence.append(0)
        else:
            count = 0
            for j in range(max(0, i-2), i+1):
                if j > 0:
                    price_dir = np.sign(data['close'].iloc[j] - data['close'].iloc[j-1])
                    vol_dir = np.sign(data['volume'].iloc[j] - data['volume'].iloc[j-1])
                    if price_dir == vol_dir:
                        count += 1
            alignment_persistence.append(count)
    data['alignment_persistence'] = alignment_persistence
    
    data['price_efficiency'] = abs(data['price_change']) / (data['daily_range'] + 1e-8)
    data['volume_efficiency'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3) + 1e-8) / 3
    data['efficiency_convergence'] = data['price_efficiency'] * data['volume_efficiency']
    
    # Opening Gap Analysis
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['gap_absorption'] = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1) + 1e-8)
    data['gap_momentum_persistence'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Medium-Term Convergence (5-10 days)
    data['price_efficiency_5d'] = (data['close'] - data['close'].shift(5)) / (abs(data['close'] - data['close'].shift(1)).rolling(window=5).sum() + 1e-8)
    data['volume_momentum_5d'] = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8)
    data['efficiency_volume_corr'] = np.sign(data['price_efficiency_5d']) * np.sign(data['volume_momentum_5d'])
    
    data['range_compression'] = data['daily_range'] / (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min() + 1e-8)
    
    # Calculate volatility persistence score
    vol_persistence = []
    for i in range(len(data)):
        if i < 4:
            vol_persistence.append(0)
        else:
            count = 0
            for j in range(max(0, i-4), i+1):
                if j > 0:
                    if data['daily_range'].iloc[j] > data['daily_range'].iloc[j-1]:
                        count += 1
            vol_persistence.append(count)
    data['vol_persistence_score'] = vol_persistence
    
    data['regime_stability'] = 1 - abs(data['daily_range'] / (data['daily_range'].shift(1) + 1e-8) - 1)
    
    # Liquidity Flow Patterns
    data['volume_cluster_momentum'] = (data['volume'] + data['volume'].shift(1)) / (data['volume'].shift(4) + data['volume'].shift(5) + 1e-8)
    data['volume_efficiency_trend'] = data['volume_per_unit_move'] / (data['volume_per_unit_move'].shift(5) + 1e-8)
    data['liquidity_acceleration'] = (data['volume'] / (data['volume'].shift(3) + 1e-8)) - (data['volume'].shift(3) / (data['volume'].shift(6) + 1e-8))
    
    # Cross-Timeframe Divergence Detection
    data['momentum_ratio'] = (data['close'] / data['close'].shift(3)) / (data['close'] / data['close'].shift(10) + 1e-8)
    data['volume_acceleration_ratio'] = (data['volume'] / data['volume'].shift(3)) / (data['volume'] / data['volume'].shift(10) + 1e-8)
    data['timeframe_convergence'] = np.sign(data['momentum_ratio'] - 1) * np.sign(data['volume_acceleration_ratio'] - 1)
    
    data['price_efficiency_div'] = data['price_efficiency'] - data['price_efficiency'].shift(3)
    data['volume_efficiency_div'] = data['volume_efficiency'] - data['volume_efficiency'].shift(3)
    data['cross_efficiency_alignment'] = np.sign(data['price_efficiency_div']) * np.sign(data['volume_efficiency_div'])
    
    data['high_vol_low_liq'] = (data['daily_range'] / data['close']) * (1 / (data['volume_per_unit_move'] + 1e-8))
    data['low_vol_high_liq'] = (1 / (data['daily_range'] / data['close'] + 1e-8)) * data['volume_per_unit_move']
    
    # Calculate mismatch persistence
    mismatch_persistence = []
    for i in range(len(data)):
        if i < 4:
            mismatch_persistence.append(0)
        else:
            count = 0
            current_sign = np.sign(data['daily_range'].iloc[i] * data['volume_per_unit_move'].iloc[i])
            for j in range(max(0, i-4), i):
                if j >= 0:
                    if np.sign(data['daily_range'].iloc[j] * data['volume_per_unit_move'].iloc[j]) == current_sign:
                        count += 1
            mismatch_persistence.append(count)
    data['mismatch_persistence'] = mismatch_persistence
    
    # Adaptive Signal Integration
    # Determine volatility regime
    data['vol_regime'] = np.where(data['daily_range'] > data['daily_range'].rolling(window=20).median(), 'high', 'low')
    
    # Regime-Weighted Component Blending
    # Base convergence components
    data['base_convergence'] = data['directional_alignment'] * data['efficiency_convergence']
    
    # Apply regime-specific transformations
    high_vol_mask = data['vol_regime'] == 'high'
    low_vol_mask = data['vol_regime'] == 'low'
    
    data['regime_adjusted_signal'] = 0
    data.loc[high_vol_mask, 'regime_adjusted_signal'] = (
        data.loc[high_vol_mask, 'base_convergence'] / (data.loc[high_vol_mask, 'daily_range'] + 1e-8) +
        data.loc[high_vol_mask, 'volume_efficiency_div'] * 0.5
    )
    data.loc[low_vol_mask, 'regime_adjusted_signal'] = (
        data.loc[low_vol_mask, 'base_convergence'] * (data.loc[low_vol_mask, 'close'] / (data.loc[low_vol_mask, 'daily_range'] + 1e-8)) +
        data.loc[low_vol_mask, 'directional_alignment'] * 0.3 +
        data.loc[low_vol_mask, 'volume_concentration'] * 0.2
    )
    
    # Persistence enhancement
    data['persistence_enhanced'] = data['regime_adjusted_signal'] * data['alignment_persistence']
    
    # Multi-Scale Confidence Scoring
    data['short_term_confidence'] = data['alignment_persistence'] * data['efficiency_convergence']
    data['medium_term_confidence'] = data['vol_persistence_score'] * data['liquidity_acceleration']
    data['cross_timeframe_confidence'] = data['timeframe_convergence'] * data['mismatch_persistence']
    
    # Dynamic Factor Generation
    # Volatility-Liquidity Convergence Score
    data['convergence_score'] = (
        data['persistence_enhanced'] * 0.4 +
        data['short_term_confidence'] * 0.3 +
        data['medium_term_confidence'] * 0.2 +
        data['cross_timeframe_confidence'] * 0.1
    )
    
    # Divergence Early Warning
    data['divergence_warning'] = (
        data['cross_efficiency_alignment'] * 0.4 +
        data['high_vol_low_liq'] * 0.3 +
        data['timeframe_convergence'] * 0.3
    )
    
    # Final Composite Factor
    data['final_factor'] = (
        data['convergence_score'] * (1 - abs(data['divergence_warning'])) -
        data['divergence_warning'] * 0.2
    )
    
    # Clean up and return
    result = data['final_factor'].fillna(0)
    return result
