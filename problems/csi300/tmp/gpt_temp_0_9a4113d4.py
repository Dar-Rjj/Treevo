import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Momentum factor
    """
    data = df.copy()
    
    # 1. Intraday Volatility Clustering
    # High-Low Range
    data['hl_range'] = data['high'] - data['low']
    
    # High-Low Range Autocorrelation (1,2,3-day lags)
    for lag in [1, 2, 3]:
        data[f'hl_range_lag{lag}'] = data['hl_range'].shift(lag)
    
    # Volume-Volatility correlation
    data['volume_vol_corr'] = data['volume'].rolling(window=5).corr(data['hl_range'])
    
    # 2. Price Path Efficiency
    # Daily efficiency: (High - Low) / (|Open - Close| + |High - Low|)
    data['abs_oc'] = abs(data['open'] - data['close'])
    data['path_efficiency'] = data['hl_range'] / (data['abs_oc'] + data['hl_range'])
    
    # Gap efficiency: |Close - Previous Close| / (High - Low)
    data['gap_efficiency'] = abs(data['close'] - data['close'].shift(1)) / (data['hl_range'] + 1e-8)
    
    # Multi-day efficiency decay (3-day weighted average)
    weights = np.array([0.5, 0.3, 0.2])
    data['eff_3day'] = 0
    for i, w in enumerate(weights):
        data['eff_3day'] += w * data['path_efficiency'].shift(i)
    
    # 3. Order Flow Imbalance Proxy
    # Close-to-Open position in range
    data['range_position'] = (data['close'] - data['open']) / (data['hl_range'] + 1e-8)
    
    # Volume-weighted range asymmetry
    data['vw_range_asym'] = (data['high'] - data['close']) / (data['hl_range'] + 1e-8) * data['volume']
    
    # Persistence of extreme positioning
    data['extreme_pos_persistence'] = (abs(data['range_position']) > 0.8).rolling(window=3).sum()
    
    # 4. Volatility Regime Classification
    # Rolling volatility (10-day)
    data['returns'] = data['close'].pct_change()
    data['volatility_10d'] = data['returns'].rolling(window=10).std()
    
    # Volatility regime (1 = high, 0 = low)
    vol_median = data['volatility_10d'].rolling(window=20).median()
    data['vol_regime'] = (data['volatility_10d'] > vol_median).astype(int)
    
    # Regime transition detection
    data['regime_change'] = data['vol_regime'].diff()
    
    # 5. Adaptive Signal Combination
    # Momentum signals with different lookbacks
    lookbacks = [3, 5, 10]
    momentum_signals = {}
    
    for lookback in lookbacks:
        # Price momentum
        momentum_signals[f'mom_{lookback}'] = data['close'] / data['close'].shift(lookback) - 1
        
        # Efficiency-adjusted momentum
        eff_weight = data['path_efficiency'].rolling(window=lookback).mean()
        momentum_signals[f'eff_mom_{lookback}'] = momentum_signals[f'mom_{lookback}'] * eff_weight
    
    # Regime-dependent weights
    high_vol_weights = {'mom_3': 0.4, 'mom_5': 0.3, 'mom_10': 0.3}
    low_vol_weights = {'mom_3': 0.2, 'mom_5': 0.3, 'mom_10': 0.5}
    
    # Dynamic lookback selection based on volatility persistence
    vol_persistence = (data['vol_regime'] == data['vol_regime'].shift(1)).rolling(window=5).mean()
    adaptive_lookback = np.where(vol_persistence > 0.7, 10, 5)
    
    # Final factor calculation
    factor = pd.Series(index=data.index, dtype=float)
    
    for date in data.index:
        if pd.isna(data.loc[date, 'vol_regime']):
            factor.loc[date] = np.nan
            continue
            
        regime = data.loc[date, 'vol_regime']
        weights = high_vol_weights if regime == 1 else low_vol_weights
        
        # Base momentum component
        mom_component = 0
        for signal, weight in weights.items():
            mom_component += weight * momentum_signals[signal].loc[date]
        
        # Efficiency adjustment
        eff_component = data.loc[date, 'path_efficiency'] * data.loc[date, 'eff_3day']
        
        # Order flow component
        of_component = (data.loc[date, 'range_position'] * 
                       data.loc[date, 'extreme_pos_persistence'])
        
        # Volatility clustering adjustment
        vol_cluster_adj = data.loc[date, 'volume_vol_corr'] * data.loc[date, 'hl_range']
        
        # Combine components with regime-adaptive weights
        if regime == 1:  # High volatility
            factor.loc[date] = (0.4 * mom_component + 
                               0.3 * eff_component + 
                               0.2 * of_component + 
                               0.1 * vol_cluster_adj)
        else:  # Low volatility
            factor.loc[date] = (0.5 * mom_component + 
                               0.25 * eff_component + 
                               0.15 * of_component + 
                               0.1 * vol_cluster_adj)
    
    return factor
