import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-period momentum calculation
    df['mom_3'] = df['close'] / df['close'].shift(3) - 1
    df['mom_8'] = df['close'] / df['close'].shift(8) - 1
    df['mom_21'] = df['close'] / df['close'].shift(21) - 1
    df['mom_55'] = df['close'] / df['close'].shift(55) - 1
    
    # Momentum divergence detection
    momentum_cols = ['mom_3', 'mom_8', 'mom_21', 'mom_55']
    df['mom_max_diff'] = np.max([np.abs(df[mom1] - df[mom2]) 
                               for i, mom1 in enumerate(momentum_cols) 
                               for j, mom2 in enumerate(momentum_cols) 
                               if i > j], axis=0)
    
    df['mom_variance'] = df[momentum_cols].var(axis=1)
    
    # Acceleration patterns
    df['mom_accel_3_8'] = df['mom_3'] - df['mom_8']
    df['mom_accel_8_21'] = df['mom_8'] - df['mom_21']
    df['mom_accel_21_55'] = df['mom_21'] - df['mom_55']
    
    # Volume dynamics
    df['vol_momentum'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    df['vol_skew'] = df['volume'].rolling(20).apply(lambda x: x.skew(), raw=False)
    df['vol_clustering'] = (df['volume'] / df['volume'].rolling(20).mean()).rolling(5).std()
    
    # Amount efficiency features
    df['amount_volume_ratio'] = df['amount'] / (df['volume'] * df['close'])
    df['amount_momentum'] = df['amount'].rolling(5).mean() / df['amount'].rolling(20).mean()
    df['amount_skew'] = df['amount'].rolling(20).apply(lambda x: x.skew(), raw=False)
    
    # Synchronization metrics
    df['price_vol_corr'] = df['close'].rolling(10).corr(df['volume'])
    df['price_amount_corr'] = df['close'].rolling(10).corr(df['amount'])
    df['vol_amount_corr'] = df['volume'].rolling(10).corr(df['amount'])
    df['corr_vs_avg'] = df['price_vol_corr'] - df['price_vol_corr'].rolling(20).mean()
    
    # Asymmetric confirmation signals
    df['vol_up_amount_down'] = ((df['vol_momentum'] > 1) & (df['amount_momentum'] < 1)).astype(int)
    df['vol_down_amount_up'] = ((df['vol_momentum'] < 1) & (df['amount_momentum'] > 1)).astype(int)
    df['vol_amount_mom_div'] = np.abs(df['vol_momentum'] - df['amount_momentum'])
    
    # Breakout confirmation
    df['mom_div_vol_conf'] = df['mom_max_diff'] * df['vol_momentum']
    df['mom_div_amount_conf'] = df['mom_max_diff'] * df['amount_momentum']
    df['breakout_sync'] = df['mom_max_diff'] * df['vol_amount_corr']
    
    # Trading intensity asymmetry
    df['vol_cluster_vs_amount'] = df['vol_clustering'] / (df['amount'].rolling(20).std() / df['amount'].rolling(20).mean())
    df['efficiency_div'] = np.abs(df['vol_momentum'] - df['amount_volume_ratio'].rolling(5).mean())
    
    # Volatility regime detection
    df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
    df['vol_regime'] = df['volatility_20d'] / df['volatility_20d'].rolling(50).mean()
    
    # Synchronization regime filtering
    df['corr_regime'] = df['price_vol_corr'].rolling(20).std()
    df['corr_change'] = df['price_vol_corr'].diff(5)
    
    # Composite weighting components
    momentum_strength = df['mom_max_diff'] * df['mom_variance']
    confirmation_level = (df['mom_div_vol_conf'] + df['mom_div_amount_conf']) / 2
    regime_scaling = 1 / (1 + np.abs(df['vol_regime'] - 1))
    
    # Multi-dimensional divergence score
    divergence_score = (df['mom_max_diff'] + 
                       np.abs(df['mom_accel_3_8']) + 
                       np.abs(df['mom_accel_8_21']) + 
                       np.abs(df['mom_accel_21_55'])) / 4
    
    # Synchronization-weighted momentum signals
    sync_weighted_momentum = divergence_score * (1 + df['price_vol_corr'])
    
    # Asymmetry-confirmed breakout patterns
    asymmetry_confirmed = (df['breakout_sync'] * 
                          (1 + df['vol_amount_mom_div']) * 
                          (1 - np.abs(df['vol_up_amount_down'] - df['vol_down_amount_up'])))
    
    # Regime-adaptive final alpha
    alpha = (divergence_score * 0.3 +
             sync_weighted_momentum * 0.25 +
             asymmetry_confirmed * 0.25 +
             momentum_strength * 0.1 +
             confirmation_level * 0.1) * regime_scaling
    
    return alpha
