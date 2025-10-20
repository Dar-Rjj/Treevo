import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Efficiency Analysis
    # Calculate price momentum for different timeframes
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_21d'] = data['close'] / data['close'].shift(21) - 1
    
    # Compute price change per unit volume
    data['price_vol_efficiency_3d'] = data['momentum_3d'] / data['volume'].rolling(3).mean()
    data['price_vol_efficiency_8d'] = data['momentum_8d'] / data['volume'].rolling(8).mean()
    data['price_vol_efficiency_21d'] = data['momentum_21d'] / data['volume'].rolling(21).mean()
    
    # Calculate volume-adjusted price range
    data['vol_adjusted_range'] = (data['high'] - data['low']) * data['volume']
    
    # Volatility Regime Classification
    # Calculate ATR
    data['tr'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    data['atr_20d'] = data['tr'].rolling(20).mean()
    data['atr_60d'] = data['tr'].rolling(60).mean()
    data['atr_ratio'] = data['atr_20d'] / data['atr_60d']
    
    # Compare rolling std dev to median volatility
    data['volatility_20d'] = data['close'].pct_change().rolling(20).std()
    data['volatility_60d_median'] = data['close'].pct_change().rolling(60).std().rolling(20).median()
    data['vol_ratio'] = data['volatility_20d'] / data['volatility_60d_median']
    
    # Classify volatility regimes
    data['vol_regime'] = 'Normal'
    data.loc[data['atr_ratio'] > 1.2, 'vol_regime'] = 'High'
    data.loc[data['atr_ratio'] < 0.8, 'vol_regime'] = 'Low'
    
    # Volume Structure & Persistence
    # Volume momentum
    data['volume_momentum_5d'] = data['volume'] / data['volume'].rolling(5).mean() - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].rolling(10).mean() - 1
    
    # Volume concentration (simplified using rolling periods)
    data['volume_30min_ratio'] = data['volume'].rolling(3).sum() / data['volume'].rolling(10).sum()
    
    # Volume autocorrelation
    data['volume_autocorr_5d'] = data['volume'].rolling(5).apply(lambda x: x.autocorr(), raw=False)
    
    # Consecutive volume increase/decrease
    data['volume_change'] = data['volume'].pct_change()
    data['volume_increase_streak'] = data['volume_change'].gt(0).groupby((data['volume_change'].gt(0) != data['volume_change'].gt(0).shift()).cumsum()).cumcount() + 1
    data['volume_decrease_streak'] = data['volume_change'].lt(0).groupby((data['volume_change'].lt(0) != data['volume_change'].lt(0).shift()).cumsum()).cumcount() + 1
    
    # Divergence Pattern Detection
    # Compare momentum signs
    data['momentum_divergence'] = np.sign(data['momentum_3d']) - np.sign(data['momentum_21d'])
    
    # Volume-adjusted range confirmation
    data['range_momentum_alignment'] = np.sign(data['vol_adjusted_range'].pct_change()) * np.sign(data['momentum_3d'])
    
    # Price-volume synchronization
    data['price_volume_sync'] = np.sign(data['momentum_3d']) * np.sign(data['volume_momentum_5d'])
    
    # Regime-Adaptive Signal Generation
    # Initialize regime-specific signals
    data['high_vol_signal'] = 0.0
    data['low_vol_signal'] = 0.0
    data['normal_vol_signal'] = 0.0
    
    # High Volatility regime signals
    high_vol_mask = data['vol_regime'] == 'High'
    data.loc[high_vol_mask, 'high_vol_signal'] = (
        data.loc[high_vol_mask, 'momentum_divergence'] * 0.3 +
        data.loc[high_vol_mask, 'volume_30min_ratio'] * 0.4 +
        data.loc[high_vol_mask, 'range_momentum_alignment'] * 0.3
    )
    
    # Low Volatility regime signals
    low_vol_mask = data['vol_regime'] == 'Low'
    data.loc[low_vol_mask, 'low_vol_signal'] = (
        data.loc[low_vol_mask, 'volume_increase_streak'] * 0.4 +
        data.loc[low_vol_mask, 'volume_autocorr_5d'] * 0.3 +
        (data['close'].rolling(20).max() / data['close'] - 1).clip(upper=0.1) * 0.3
    )
    
    # Normal Volatility regime signals
    normal_vol_mask = data['vol_regime'] == 'Normal'
    data.loc[normal_vol_mask, 'normal_vol_signal'] = (
        data.loc[normal_vol_mask, 'price_vol_efficiency_8d'] * 0.4 +
        data.loc[normal_vol_mask, 'price_volume_sync'] * 0.3 +
        data.loc[normal_vol_mask, 'volume_momentum_10d'] * 0.3
    )
    
    # Volume clustering analysis for confirmation
    data['volume_cluster_strength'] = (
        data['volume_autocorr_5d'].fillna(0) * 0.5 +
        (data['volume_increase_streak'] / 10).clip(upper=1) * 0.3 +
        data['volume_30min_ratio'] * 0.2
    )
    
    # Composite Alpha Output
    # Combine regime-weighted components
    regime_signals = pd.DataFrame({
        'High': data['high_vol_signal'],
        'Low': data['low_vol_signal'], 
        'Normal': data['normal_vol_signal']
    })
    
    data['regime_weighted_signal'] = regime_signals.lookup(data.index, data['vol_regime'])
    
    # Final efficiency regime factor
    data['alpha_factor'] = (
        data['regime_weighted_signal'] * 0.6 +
        data['volume_cluster_strength'] * 0.2 +
        data['price_vol_efficiency_21d'].fillna(0) * 0.2
    )
    
    # Normalize the final factor
    alpha_series = data['alpha_factor'].fillna(0)
    
    return alpha_series
