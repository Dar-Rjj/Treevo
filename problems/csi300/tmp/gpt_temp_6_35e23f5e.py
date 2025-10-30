import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Price-Volume Synchronization with Momentum Confirmation
    """
    data = df.copy()
    
    # Price-Volume Synchronization Analysis
    # Directional Synchronization
    data['price_dir'] = np.sign(data['close'] - data['close'].shift(1))
    data['volume_dir'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['dir_match'] = (data['price_dir'] == data['volume_dir']).astype(int)
    
    # 3-day synchronization persistence
    data['sync_persistence'] = data['dir_match'].rolling(window=3, min_periods=1).sum()
    
    # Synchronization strength
    data['price_change_pct'] = (data['close'] - data['close'].shift(1)).abs() / data['close'].shift(1)
    data['volume_change_pct'] = (data['volume'] - data['volume'].shift(1)).abs() / data['volume'].shift(1).replace(0, 1)
    data['sync_strength'] = data['price_change_pct'] * data['volume_change_pct']
    
    # Magnitude Synchronization
    data['price_volume_ratio'] = (data['price_change_pct'] + 1e-8) / (data['volume'] / data['volume'].shift(1).replace(0, 1) + 1e-8)
    data['pv_ratio_5d_avg'] = data['price_volume_ratio'].rolling(window=5, min_periods=1).mean()
    data['pv_ratio_deviation'] = data['price_volume_ratio'] / (data['pv_ratio_5d_avg'] + 1e-8)
    
    # Volume-price elasticity (5-day correlation)
    data['abs_price_ret'] = (data['close'] - data['close'].shift(1)).abs() / data['close'].shift(1)
    data['volume_elasticity'] = data['abs_price_ret'].rolling(window=5, min_periods=1).corr(data['volume'])
    
    # Momentum Confirmation Framework
    # Short-term Momentum Components
    data['momentum_1d'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['intraday_momentum'] = (data['close'] - data['open']) / ((data['high'] - data['low']).replace(0, 1e-8))
    
    # Volume-Weighted Momentum
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_momentum'] = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    
    data['volume_10d_avg'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_surge'] = (data['volume'] > 1.5 * data['volume_10d_avg']).astype(int)
    
    # 3-day volume slope
    data['volume_slope'] = data['volume'].rolling(window=3, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (len(x) - 1) if len(x) > 1 else 0
    )
    
    # Market Regime Classification
    # Volatility Regime Detection
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        )
    )
    data['atr_5d'] = data['tr'].rolling(window=5, min_periods=1).mean()
    data['atr_20d_median'] = data['tr'].rolling(window=20, min_periods=1).median()
    data['high_vol_regime'] = (data['atr_5d'] > data['atr_20d_median']).astype(int)
    
    data['price_range_5d'] = (data['high'].rolling(window=5, min_periods=1).max() - 
                             data['low'].rolling(window=5, min_periods=1).min())
    data['price_stability'] = data['price_range_5d'] / data['close']
    
    # Synchronization Regime Analysis
    data['sync_level_5d'] = data['dir_match'].rolling(window=5, min_periods=1).mean()
    data['pv_ratio_var'] = data['price_volume_ratio'].rolling(window=5, min_periods=1).var()
    
    # Trend persistence
    data['price_dir_shift'] = data['price_dir'].shift(1)
    data['same_dir'] = (data['price_dir'] == data['price_dir_shift']).astype(int)
    data['trend_persistence'] = data['same_dir'].rolling(window=5, min_periods=1).apply(
        lambda x: x[::-1].cumprod().sum() if len(x) > 0 else 0
    )
    
    # Regime-Adaptive Signal Generation
    # High Synchronization Regime Processing
    data['high_sync_regime'] = (data['sync_level_5d'] > 0.7).astype(int)
    
    # Strong synchronization with momentum confirmation
    data['strong_sync_momentum'] = (
        (data['high_sync_regime'] == 1) & 
        (data['trend_persistence'] > 2) &
        (data['volume_surge'] == 1)
    ).astype(int)
    
    # Synchronization breakdown detection
    data['sync_strength_3d_avg'] = data['sync_strength'].rolling(window=3, min_periods=1).mean()
    data['sync_breakdown'] = (
        (data['sync_strength'] < 0.8 * data['sync_strength_3d_avg']) &
        (data['high_sync_regime'] == 1)
    ).astype(int)
    
    # Low Synchronization Regime Processing
    data['low_sync_regime'] = (data['sync_level_5d'] < 0.3).astype(int)
    
    # Price-volume divergence patterns
    data['price_volume_divergence'] = (
        (data['price_dir'] != data['volume_dir']) & 
        (data['low_sync_regime'] == 1)
    ).astype(int)
    
    # Regime transition signals
    data['sync_strength_change'] = data['sync_strength'] - data['sync_strength'].shift(1)
    data['regime_transition'] = (
        (data['sync_strength_change'].abs() > data['sync_strength_change'].rolling(window=10, min_periods=1).std()) &
        (data['high_vol_regime'].diff() != 0)
    ).astype(int)
    
    # Composite Alpha Factor Construction
    # Synchronization-Momentum Integration
    data['sync_momentum_base'] = (
        data['sync_strength'] * data['momentum_1d'] * 
        np.sign(data['volume_momentum']) * data['intraday_momentum']
    )
    
    # Apply volume confirmation weighting
    data['volume_confirmation'] = data['volume_momentum'] * data['volume_surge']
    data['weighted_sync_momentum'] = data['sync_momentum_base'] * (1 + 0.5 * data['volume_confirmation'])
    
    # Incorporate regime-specific scaling
    data['regime_scaling'] = np.where(
        data['high_sync_regime'] == 1,
        1.2,  # Enhance in high synchronization
        np.where(
            data['low_sync_regime'] == 1,
            0.8,  # Reduce in low synchronization
            1.0   # Neutral in transition
        )
    )
    
    # Adaptive Filtering Framework
    # Volatility-based signal adjustment
    data['volatility_adjustment'] = np.where(
        data['high_vol_regime'] == 1,
        0.7,  # Reduce in high volatility
        1.3   # Enhance in low volatility
    )
    
    # Liquidity efficiency assessment
    data['price_impact'] = (data['high'] - data['low']) / data['close']
    data['volume_amount_ratio'] = data['volume'] / (data['amount'].replace(0, 1e-8))
    
    # Apply minimum efficiency threshold
    data['liquidity_efficiency'] = np.where(
        data['price_impact'] < 0.05,
        1.1,  # Good liquidity
        np.where(
            data['price_impact'] > 0.1,
            0.6,  # Poor liquidity
            0.9   # Average liquidity
        )
    )
    
    # Final Signal Output
    data['composite_alpha'] = (
        data['weighted_sync_momentum'] * 
        data['regime_scaling'] * 
        data['volatility_adjustment'] * 
        data['liquidity_efficiency'] *
        (1 + 0.2 * data['strong_sync_momentum']) *
        (1 - 0.3 * data['sync_breakdown']) *
        (1 + 0.1 * data['regime_transition'])
    )
    
    # Clean up and return
    alpha_series = data['composite_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_series
