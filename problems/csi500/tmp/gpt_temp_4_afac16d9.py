import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Momentum with Microstructure Noise Filtering
    """
    data = df.copy()
    
    # Multi-Scale Volatility Regime Classification
    # Intraday volatility estimation
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['intraday_vol_5d'] = data['intraday_range'].rolling(window=5, min_periods=3).std()
    
    # Overnight gap volatility
    data['overnight_gap'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['overnight_vol_10d'] = data['overnight_gap'].rolling(window=10, min_periods=5).mean()
    
    # Volatility regime assignment
    intraday_vol_median_20d = data['intraday_range'].rolling(window=20, min_periods=10).median()
    data['vol_regime'] = 'normal'
    data.loc[data['intraday_range'] > 1.5 * intraday_vol_median_20d, 'vol_regime'] = 'high'
    data.loc[data['intraday_range'] < 0.7 * intraday_vol_median_20d, 'vol_regime'] = 'low'
    
    # Microstructure Noise Identification
    # Bid-ask bounce detection
    data['return_sign'] = np.sign(data['close'] - data['close'].shift(1))
    data['sign_change'] = data['return_sign'] * data['return_sign'].shift(1)
    data['bounce_count'] = 0
    for i in range(2, len(data)):
        if data['sign_change'].iloc[i] == 1:
            data.loc[data.index[i], 'bounce_count'] = data['bounce_count'].iloc[i-1] + 1
    
    # Price discreteness analysis
    data['price_mod'] = (data['close'] * 100).astype(int) % 100
    data['tick_size_freq'] = data['price_mod'].rolling(window=10, min_periods=5).apply(
        lambda x: (x == 0).sum() / len(x) if len(x) > 0 else 0
    )
    
    # Volume-price divergence
    data['price_change_pct'] = abs(data['close'] / data['close'].shift(1) - 1)
    data['volume_change_ratio'] = data['volume'] / data['volume'].shift(1)
    data['vp_ratio'] = data['price_change_pct'] / np.maximum(data['volume_change_ratio'], 0.001)
    data['vp_ratio_10d_avg'] = data['vp_ratio'].rolling(window=10, min_periods=5).mean()
    
    # Flag suspicious microstructure patterns
    data['microstructure_noise'] = 0
    data.loc[data['bounce_count'] >= 2, 'microstructure_noise'] += 1
    data.loc[data['tick_size_freq'] > 0.3, 'microstructure_noise'] += 1
    data.loc[data['vp_ratio'] > 2 * data['vp_ratio_10d_avg'], 'microstructure_noise'] += 1
    
    # Regime-Adaptive Momentum Construction
    # Volatility-scaled momentum signals
    data['momentum_2d'] = (data['close'] / data['close'].shift(2) - 1) / np.maximum(data['intraday_range'], 0.001)
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = (data['close'] / data['close'].shift(10) - 1) * 1.5
    
    # Noise-filtered momentum adjustment
    data['regime_momentum'] = data['momentum_5d']  # default normal vol
    
    # High volatility regime
    high_vol_mask = data['vol_regime'] == 'high'
    data.loc[high_vol_mask, 'regime_momentum'] = data.loc[high_vol_mask, 'momentum_2d']
    
    # Low volatility regime  
    low_vol_mask = data['vol_regime'] == 'low'
    data.loc[low_vol_mask, 'regime_momentum'] = data.loc[low_vol_mask, 'momentum_10d']
    
    # Noise adjustment
    noise_mask = data['microstructure_noise'] >= 2
    data.loc[noise_mask & high_vol_mask, 'regime_momentum'] = data.loc[noise_mask & high_vol_mask, 'momentum_5d']
    data.loc[noise_mask & (data['vol_regime'] == 'normal'), 'regime_momentum'] = (
        data.loc[noise_mask & (data['vol_regime'] == 'normal'), 'momentum_10d']
    )
    
    # Weight reduction for noisy periods
    data.loc[data['microstructure_noise'] >= 1, 'regime_momentum'] *= 0.8
    data.loc[data['microstructure_noise'] >= 3, 'regime_momentum'] *= 0.5
    
    # Volume-Confirmed Price Action
    # Volume trend alignment
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ratio'] = data['volume'] / data['volume_5d_avg']
    data['volume_20d_median'] = data['volume'].rolling(window=20, min_periods=10).median()
    
    # Volume clustering analysis
    data['high_volume'] = data['volume'] > 1.2 * data['volume_20d_median']
    data['volume_persistence'] = data['high_volume'].rolling(window=5, min_periods=3).sum()
    
    # Volume-momentum confirmation
    data['volume_confirmation'] = 1.0  # neutral
    strong_conf_mask = (data['volume_ratio'] > 1.2) & (abs(data['regime_momentum']) > 0.02)
    divergence_mask = (data['volume_ratio'] < 0.8) & (abs(data['regime_momentum']) > 0.03)
    
    data.loc[strong_conf_mask, 'volume_confirmation'] = 1.3
    data.loc[divergence_mask, 'volume_confirmation'] = 0.7
    
    # Adaptive Factor Integration
    # Regime persistence adjustment
    data['regime_persistence'] = 1.0
    data['regime_change'] = data['vol_regime'] != data['vol_regime'].shift(1)
    data['same_regime_count'] = 0
    
    for i in range(1, len(data)):
        if not data['regime_change'].iloc[i]:
            data.loc[data.index[i], 'same_regime_count'] = data['same_regime_count'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'same_regime_count'] = 1
    
    data.loc[data['same_regime_count'] >= 3, 'regime_persistence'] = 1.2
    data.loc[data['regime_change'], 'regime_persistence'] = 0.9
    
    # Special handling for high to low volatility transition
    high_to_low = (data['vol_regime'].shift(1) == 'high') & (data['vol_regime'] == 'low')
    data.loc[high_to_low, 'regime_persistence'] = 0.8
    
    # Dynamic Factor Output
    data['core_factor'] = data['regime_momentum'] * data['volume_confirmation'] * data['regime_persistence']
    
    # Microstructure quality filter
    data['final_factor'] = data['core_factor']
    
    # Apply quality filter
    high_noise_mask = data['microstructure_noise'] >= 3
    data.loc[high_noise_mask, 'final_factor'] = data['core_factor'].shift(1) * 0.9
    
    # Minimum data quality requirement
    min_data_mask = (
        data['intraday_range'].notna() & 
        data['volume'].notna() & 
        (data.index >= data.index[10])  # ensure enough history
    )
    data.loc[~min_data_mask, 'final_factor'] = np.nan
    
    return data['final_factor']
