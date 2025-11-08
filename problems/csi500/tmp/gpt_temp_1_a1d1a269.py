import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume Synchronization with Regime-Dependent Acceleration
    """
    data = df.copy()
    
    # Calculate Price-Volume Synchronization Patterns
    # Short-term price-volume alignment
    data['price_return_3d'] = data['close'].pct_change(3)
    data['volume_change_3d'] = data['volume'].pct_change(3)
    
    # Rolling price-volume correlation
    data['pv_corr_5d'] = data['close'].pct_change().rolling(5).corr(data['volume'].pct_change())
    
    # Rolling divergence detection
    data['pv_divergence'] = np.abs(data['price_return_3d'] - data['volume_change_3d'])
    data['sync_break_short'] = data['pv_divergence'].rolling(3).std()
    
    # Medium-term price-volume dynamics
    data['price_momentum_10d'] = data['close'].pct_change(10)
    data['volume_momentum_10d'] = data['volume'].pct_change(10)
    data['pv_acceleration_alignment'] = (data['price_momentum_10d'].pct_change(3) * 
                                       data['volume_momentum_10d'].pct_change(3))
    
    # Synchronization strength quantification
    data['directional_alignment'] = np.sign(data['price_return_3d']) * np.sign(data['volume_change_3d'])
    data['magnitude_matching'] = 1 - (np.abs(data['price_return_3d'] - data['volume_change_3d']) / 
                                    (np.abs(data['price_return_3d']) + np.abs(data['volume_change_3d']) + 1e-8))
    data['sync_persistence'] = data['directional_alignment'].rolling(5).mean()
    
    # Assess Multi-Timeframe Regime Dynamics
    # Price regime characteristics
    data['price_efficiency_5d'] = (data['close'] - data['close'].shift(5)).abs() / (
        data['high'].rolling(5).max() - data['low'].rolling(5).min() + 1e-8)
    
    data['directional_consistency_10d'] = (data['close'].pct_change().rolling(10).apply(
        lambda x: np.sum(x > 0) / len(x) if len(x) == 10 else np.nan))
    
    # Fractal dimension approximation using price range
    data['fractal_dim'] = np.log(data['high'].rolling(5).max() - data['low'].rolling(5).min() + 1) / np.log(5)
    
    # Volume regime transitions
    data['volume_clustering'] = data['volume'].rolling(5).std() / (data['volume'].rolling(5).mean() + 1e-8)
    data['volume_persistence'] = (data['volume'].pct_change().rolling(5).apply(
        lambda x: np.sum(np.abs(x) > np.std(x)) / len(x) if len(x) == 5 else np.nan))
    data['vol_vol_ratio'] = data['volume'].pct_change().rolling(5).std() / (
        data['close'].pct_change().rolling(5).std() + 1e-8)
    
    # Regime classification using simplified clustering
    data['trend_strength'] = data['price_momentum_10d'].abs()
    data['volatility_regime'] = data['close'].pct_change().rolling(10).std()
    
    # Regime-dependent weights
    data['regime_weight_trend'] = np.where(data['trend_strength'] > data['trend_strength'].rolling(20).quantile(0.7), 
                                          data['directional_alignment'], 0)
    data['regime_weight_meanrev'] = np.where(data['trend_strength'] < data['trend_strength'].rolling(20).quantile(0.3), 
                                           data['magnitude_matching'], 0)
    data['regime_weight_volatile'] = np.where(data['volatility_regime'] > data['volatility_regime'].rolling(20).quantile(0.7), 
                                            data['sync_break_short'], 0)
    
    # Incorporate Acceleration-Convergence Dynamics
    # Price acceleration patterns
    data['price_accel_3d'] = data['price_momentum_10d'].pct_change(3)
    data['price_accel_5d'] = data['close'].pct_change(8).pct_change(5)
    data['accel_consistency'] = np.sign(data['price_accel_3d']) * np.sign(data['price_accel_5d'])
    
    # Volume acceleration convergence
    data['volume_accel_3d'] = data['volume_momentum_10d'].pct_change(3)
    data['volume_accel_5d'] = data['volume'].pct_change(8).pct_change(5)
    data['accel_convergence'] = data['price_accel_3d'] * data['volume_accel_3d']
    
    # Enhanced synchronization with acceleration
    data['sync_short_accel'] = data['pv_corr_5d'] * data['price_accel_3d']
    data['sync_medium_accel'] = data['pv_acceleration_alignment'] * data['volume_accel_5d']
    
    # Regime-dependent acceleration weights
    data['accel_weight_trend'] = np.where(data['trend_strength'] > data['trend_strength'].rolling(20).quantile(0.7), 
                                        data['price_accel_3d'], 1)
    data['accel_weight_meanrev'] = np.where(data['trend_strength'] < data['trend_strength'].rolling(20).quantile(0.3), 
                                          data['volume_accel_5d'], 1)
    
    # Detect Synchronized Breakout Conditions
    # Synchronization breakout strength
    data['sync_momentum_5d'] = data['pv_corr_5d'].pct_change(5)
    data['sync_breakout_strength'] = (data['pv_corr_5d'] * data['sync_momentum_5d'] * 
                                    data['accel_convergence'])
    
    # Regime-adaptive breakout signals
    data['breakout_threshold'] = data['sync_breakout_strength'].rolling(20).std()
    data['confirmed_breakout'] = np.where(
        np.abs(data['sync_breakout_strength']) > data['breakout_threshold'],
        data['sync_breakout_strength'] * data['volume_momentum_10d'], 0)
    
    # Multi-timeframe breakout alignment
    data['breakout_alignment'] = (data['confirmed_breakout'] * 
                                data['sync_persistence'] * 
                                data['accel_consistency'])
    
    # Combine Components with Dynamic Synchronization
    # Composite synchronization strength
    data['composite_sync'] = (
        data['sync_short_accel'] * data['accel_convergence'] * 
        data['regime_weight_trend'] * data['regime_weight_meanrev'] * 
        (1 + data['regime_weight_volatile']) * 
        data['breakout_alignment']
    )
    
    # Directional probability weights with non-linear response
    data['sync_strength'] = np.abs(data['composite_sync'])
    data['non_linear_response'] = np.tanh(data['sync_strength'] * 2)
    
    # Final Alpha Factor Construction
    alpha_factor = (
        data['composite_sync'] * 
        data['non_linear_response'] * 
        np.sign(data['price_momentum_10d']) *
        (1 + data['sync_persistence'])
    )
    
    return alpha_factor
