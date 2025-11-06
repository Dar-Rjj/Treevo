import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Volatility-Adaptive Compression-Gap Efficiency with Liquidity Confirmation
    """
    data = df.copy()
    
    # Historical volatility calculation (10-day rolling std of Close-to-Close returns)
    data['returns'] = data['close'].pct_change()
    data['hist_vol'] = data['returns'].rolling(window=10, min_periods=5).std()
    
    # Intraday volatility and regime ratio
    data['intraday_vol'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['median_intraday_vol_20d'] = data['intraday_vol'].rolling(window=20, min_periods=10).median()
    data['vol_regime_ratio'] = data['intraday_vol'] / data['median_intraday_vol_20d']
    
    # Multi-Scale Gap-Compression Integration
    # Volatility-Weighted Gap Components
    data['gap_magnitude_vol_adj'] = np.abs(data['open'] - data['close'].shift(1)) / (data['close'].shift(1) * data['hist_vol'].shift(1))
    data['gap_direction_momentum'] = np.sign(data['open'] - data['close'].shift(1)) * (data['close'] - data['open'])
    data['vol_scaled_gap_persistence'] = (data['close'] - data['open']) / data['hist_vol'].shift(1)
    
    # Multi-Period Compression Analysis
    data['range_compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['volatility_compression'] = data['close'].rolling(window=5).std() / data['close'].rolling(window=5).std().shift(5)
    
    # Midpoint-based compression
    midpoint = (data['high'] + data['low']) / 2
    data['midpoint_compression'] = ((data['close'] - midpoint) / midpoint).rolling(window=3).std()
    
    # Gap-Compression Efficiency Framework
    true_range = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hc': np.abs(data['high'] - data['close'].shift(1)),
        'lc': np.abs(data['low'] - data['close'].shift(1))
    }).max(axis=1)
    
    data['gap_efficiency'] = (data['close'] - data['open']) / np.maximum(true_range, 0.0001) * data['range_compression']
    data['vol_scaled_compression_efficiency'] = data['gap_efficiency'] * data['volatility_compression']
    
    # Asymmetric Liquidity-Gap Confirmation
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean().shift(1)
    
    # Directional Volume-Gap Synchronization
    upside_mask = data['close'] > data['open']
    downside_mask = data['close'] < data['open']
    
    data['upside_gap_confirmation'] = np.where(upside_mask, 
        data['volume_ratio'] * (data['close'] - data['open']), 0)
    data['downside_gap_absorption'] = np.where(downside_mask, 
        data['volume_ratio'] * (data['open'] - data['close']), 0)
    data['volume_weighted_gap_persistence'] = data['gap_direction_momentum'] * data['volume_ratio']
    
    # Amount-Pressure Gap Intensity
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    data['amount_pressure'] = data['amount'] / (data['volume'] * typical_price)
    
    data['buying_pressure_gap_confirmation'] = np.where(upside_mask,
        data['amount_pressure'] * (data['close'] - data['open']), 0)
    data['selling_pressure_gap_validation'] = np.where(downside_mask,
        data['amount_pressure'] * (data['open'] - data['close']), 0)
    
    # Pressure accumulation with gap alignment
    pressure_accumulation = []
    for i in range(len(data)):
        if i < 4:
            pressure_accumulation.append(np.nan)
            continue
        window_data = data.iloc[i-4:i+1]
        pressure_sum = 0
        for j in range(5):
            if j > 0:  # Skip current day for shift
                gap_dir = np.sign(window_data.iloc[j]['open'] - window_data.iloc[j-1]['close'])
                pressure_val = (np.abs(window_data.iloc[j]['close'] - window_data.iloc[j]['open']) / 
                              (window_data.iloc[j]['high'] - window_data.iloc[j]['low'])) * gap_dir
                pressure_sum += pressure_val
        pressure_accumulation.append(pressure_sum)
    
    data['pressure_accumulation'] = pressure_accumulation
    
    # Volatility-Regime Adaptive Compression Divergence
    data['pressure_compression_divergence'] = data['gap_efficiency'] * data['range_compression'] * np.sign(data['open'] - data['close'].shift(1))
    data['liquidity_confirmed_divergence'] = data['pressure_compression_divergence'] * data['volume_ratio']
    data['vol_adjusted_divergence'] = data['liquidity_confirmed_divergence'] / data['hist_vol'].shift(1)
    
    # Range-Compression Integration with Gap Alignment
    data['true_range'] = true_range
    data['range_expansion_ratio'] = data['true_range'] / data['true_range'].rolling(window=5, min_periods=3).mean().shift(1)
    data['range_compression_alignment'] = data['range_expansion_ratio'] * data['range_compression']
    data['compression_weighted_range_efficiency'] = data['gap_efficiency'] * data['range_compression_alignment']
    data['vol_scaled_range_gap'] = data['compression_weighted_range_efficiency'] / data['hist_vol'].shift(1)
    data['liquidity_confirmed_range_patterns'] = data['vol_scaled_range_gap'] * data['volume_ratio']
    
    # Multi-Scale Momentum-Compression Divergence
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['compression_adjusted_momentum'] = data['momentum_3d'] * data['volatility_compression']
    
    # Adaptive Composite Alpha Construction
    # High volatility regime signals
    high_vol_mask = data['vol_regime_ratio'] > 1.0
    low_vol_mask = data['vol_regime_ratio'] <= 1.0
    
    # High volatility regime components
    high_vol_compression_break = data['gap_efficiency'] * data['range_compression']
    high_vol_divergence = data['vol_adjusted_divergence'] * data['vol_regime_ratio']
    high_vol_range_expansion = data['compression_weighted_range_efficiency'] * data['range_expansion_ratio']
    
    # Low volatility regime components
    low_vol_liquidity_confirmed = (data['upside_gap_confirmation'] + data['downside_gap_absorption']) * data['range_compression']
    low_vol_pressure_accumulation = data['pressure_accumulation'] * data['volatility_compression']
    low_vol_momentum_filtered = data['gap_efficiency'] * (1 - np.abs(data['momentum_3d'])) * data['volatility_compression']
    
    # Combine regime-specific signals
    high_vol_composite = (high_vol_compression_break * 0.4 + 
                         high_vol_divergence * 0.4 + 
                         high_vol_range_expansion * 0.2)
    
    low_vol_composite = (low_vol_liquidity_confirmed * 0.4 + 
                        low_vol_pressure_accumulation * 0.3 + 
                        low_vol_momentum_filtered * 0.3)
    
    # Final adaptive composite alpha
    data['alpha'] = np.where(high_vol_mask, high_vol_composite, 
                           np.where(low_vol_mask, low_vol_composite, 0))
    
    # Multi-dimensional confirmation framework
    primary_component = data['gap_efficiency'] * 0.3
    liquidity_component = (data['volume_weighted_gap_persistence'] + data['buying_pressure_gap_confirmation'] + 
                          data['selling_pressure_gap_validation']) * 0.25
    range_component = data['liquidity_confirmed_range_patterns'] * 0.2
    momentum_component = data['compression_adjusted_momentum'] * data['gap_efficiency'] * 0.15
    regime_component = data['alpha'] * 0.1
    
    # Final alpha factor
    final_alpha = (primary_component + liquidity_component + range_component + 
                  momentum_component + regime_component)
    
    return final_alpha
