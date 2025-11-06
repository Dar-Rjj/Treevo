import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['pre_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['pre_close'])
    data['tr3'] = abs(data['low'] - data['pre_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility Regime Classification
    data['vol_5d_avg'] = data['true_range'].rolling(window=5).mean()
    data['vol_20d_median'] = data['true_range'].rolling(window=20).median()
    
    high_vol_threshold = 1.3 * data['vol_20d_median']
    low_vol_threshold = 0.8 * data['vol_20d_median']
    
    data['vol_regime'] = 'normal'
    data.loc[data['vol_5d_avg'] > high_vol_threshold, 'vol_regime'] = 'high'
    data.loc[data['vol_5d_avg'] < low_vol_threshold, 'vol_regime'] = 'low'
    
    # Fractal Efficiency Momentum Analysis
    data['abs_price_change_10d'] = abs(data['close'] - data['close'].shift(10))
    data['sum_tr_10d'] = data['true_range'].rolling(window=10).sum()
    data['net_efficiency'] = data['abs_price_change_10d'] / data['sum_tr_10d']
    
    # Efficiency Momentum Acceleration
    data['momentum_6d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_3d'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['momentum_acceleration'] = data['momentum_3d'] - data['momentum_6d']
    
    # Breakout Pressure with Volume Confirmation
    data['high_5d_roll'] = data['high'].rolling(window=5).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['low_5d_roll'] = data['low'].rolling(window=5).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    
    data['upper_breakout'] = (data['close'] > data['high_5d_roll']).astype(int)
    data['lower_breakout'] = (data['close'] < data['low_5d_roll']).astype(int)
    
    data['upper_breakout_vol'] = data['upper_breakout'] * data['volume']
    data['lower_breakout_vol'] = data['lower_breakout'] * data['volume']
    
    data['net_breakout_vol_5d'] = (data['upper_breakout_vol'] - data['lower_breakout_vol']).rolling(window=5).sum()
    
    # Volume Breakout Filter
    data['vol_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['volume_breakout'] = (data['volume'] > 1.2 * data['vol_10d_avg']).astype(int)
    data['volume_breakout_multiplier'] = data['volume_breakout'] * 0.8 + 1.0
    
    # Volume-Price Consistency
    data['returns'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    data['vol_price_corr_6d'] = data['returns'].rolling(window=6).corr(data['volume_change'])
    data['vol_price_corr_6d'] = data['vol_price_corr_6d'].fillna(0)
    
    data['vwap_range'] = (data['high'] - data['low']) * data['volume']
    data['avg_range'] = data['high'] - data['low']
    data['volume_range_alignment'] = (data['vwap_range'].rolling(window=6).mean() / 
                                    data['avg_range'].rolling(window=6).mean()) / data['volume'].rolling(window=6).mean()
    
    data['volume_price_consistency'] = data['vol_price_corr_6d'] * data['volume_range_alignment']
    
    # Regime-Adaptive Signal Construction
    data['core_factor'] = data['momentum_acceleration'] * data['volume_price_consistency']
    data['breakout_weighted'] = data['core_factor'] * data['net_breakout_vol_5d'] * data['volume_breakout_multiplier']
    
    # Regime-Specific Scaling
    regime_scaling = {
        'high': 0.5,
        'normal': 1.0,
        'low': 2.0
    }
    
    data['regime_scaling'] = data['vol_regime'].map(regime_scaling)
    data['regime_scaled_signal'] = data['breakout_weighted'] * data['regime_scaling']
    
    # Fractal Divergence Enhancement
    data['efficiency_momentum'] = data['net_efficiency'] * data['momentum_acceleration']
    data['divergence_corr_8d'] = data['efficiency_momentum'].rolling(window=8).corr(data['breakout_weighted'])
    data['divergence_corr_8d'] = data['divergence_corr_8d'].fillna(0)
    
    data['divergence_adjusted'] = data['regime_scaled_signal'] * data['divergence_corr_8d']
    
    # Volume Surge Multiplier Application
    data['volume_ratio'] = data['volume'] / data['vol_10d_avg']
    data['volume_surge_multiplier'] = np.clip(data['volume_ratio'], 0.5, 2.0)
    
    data['volume_adjusted_signal'] = data['divergence_adjusted'] * data['volume_surge_multiplier']
    
    # Momentum Persistence Filter
    data['signal_3d_persistence'] = data['volume_adjusted_signal'].rolling(window=3).apply(
        lambda x: len([i for i in range(1, len(x)) if x[i] * x[i-1] > 0]) / (len(x)-1) if len(x) > 1 else 0
    )
    data['persistence_strength'] = 1 + data['signal_3d_persistence']
    
    data['persistence_filtered'] = data['volume_adjusted_signal'] * data['persistence_strength']
    
    # Price Range Efficiency Normalization
    data['price_range_6d'] = (data['high'] - data['low']).rolling(window=6).sum()
    data['final_alpha'] = data['persistence_filtered'] / data['price_range_6d']
    
    # Clean up and return
    result = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
