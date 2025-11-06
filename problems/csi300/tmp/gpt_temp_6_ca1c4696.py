import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate basic price metrics
    data['returns'] = data['close'] / data['close'].shift(1) - 1
    data['range'] = data['high'] - data['low']
    data['prev_close'] = data['close'].shift(1)
    
    # Volatility Regime Classification
    # Dual Volatility Measures
    data['returns_vol_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    data['range_median_20d'] = data['range'].rolling(window=20, min_periods=10).median()
    data['range_avg_5d'] = data['range'].rolling(window=5, min_periods=3).mean()
    
    # Volatility percentiles
    data['returns_vol_percentile'] = data['returns_vol_20d'].rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) if len(x.dropna()) > 0 else 0, raw=False
    )
    
    # Regime classification
    conditions = [
        (data['returns_vol_percentile'] == 1) & (data['range_avg_5d'] > 1.3 * data['range_median_20d']),
        (data['returns_vol_percentile'] == 0) & (data['range_avg_5d'] < 0.8 * data['range_median_20d'])
    ]
    choices = ['high', 'low']
    data['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Multi-Scale Momentum-Efficiency Analysis
    # Regime-Adaptive Momentum
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_15d'] = data['close'] / data['close'].shift(15) - 1
    
    def calc_efficiency(close_series, window):
        net_move = abs(close_series - close_series.shift(window-1))
        total_move = sum([abs(close_series.shift(i) - close_series.shift(i+1)) for i in range(window-1)])
        return net_move / total_move.replace(0, np.nan)
    
    data['efficiency_3d'] = calc_efficiency(data['close'], 3)
    data['efficiency_5d'] = calc_efficiency(data['close'], 5)
    data['efficiency_8d'] = calc_efficiency(data['close'], 8)
    
    # Regime-adaptive weighting
    momentum_weights = {
        'high': (0.8, 0.2),
        'low': (0.2, 0.8),
        'normal': (0.5, 0.5)
    }
    
    efficiency_weights = {
        'high': (0.7, 0.15, 0.15),
        'low': (0.15, 0.15, 0.7),
        'normal': (1/3, 1/3, 1/3)
    }
    
    data['regime_momentum'] = np.nan
    data['weighted_efficiency'] = np.nan
    
    for regime in ['high', 'low', 'normal']:
        mask = data['vol_regime'] == regime
        w1, w2 = momentum_weights[regime]
        data.loc[mask, 'regime_momentum'] = (
            w1 * data.loc[mask, 'momentum_5d'] + w2 * data.loc[mask, 'momentum_15d']
        )
        
        w3, w4, w5 = efficiency_weights[regime]
        data.loc[mask, 'weighted_efficiency'] = (
            w3 * data.loc[mask, 'efficiency_3d'] + 
            w4 * data.loc[mask, 'efficiency_5d'] + 
            w5 * data.loc[mask, 'efficiency_8d']
        )
    
    # Volume-Pressure Alignment Analysis
    # Buy-Sell Pressure Ratio
    data['is_up_day'] = (data['close'] > data['prev_close']).astype(int)
    data['is_down_day'] = (data['close'] < data['prev_close']).astype(int)
    
    data['up_volume_10d'] = (
        data['volume'] * data['is_up_day']
    ).rolling(window=10, min_periods=5).sum()
    
    data['down_volume_10d'] = (
        data['volume'] * data['is_down_day']
    ).rolling(window=10, min_periods=5).sum()
    
    data['pressure_ratio'] = data['up_volume_10d'] / data['down_volume_10d'].replace(0, np.nan)
    
    # Volume Acceleration
    data['volume_roc'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_pressure_asymmetry'] = data['pressure_ratio'] * data['volume_roc']
    
    # Volume-Range Consistency
    data['volume_range_product'] = data['volume'] * data['range']
    data['vw_range_5d'] = (
        data['volume_range_product'].rolling(window=5, min_periods=3).sum() / 
        data['volume'].rolling(window=5, min_periods=3).sum()
    )
    data['avg_range_5d'] = data['range'].rolling(window=5, min_periods=3).mean()
    
    data['range_alignment'] = 1 - (abs(data['vw_range_5d'] - data['avg_range_5d']) / data['avg_range_5d'].replace(0, np.nan))
    
    # Transition Signal Detection
    # Volatility Breakout
    data['volatility_5d'] = data['returns'].rolling(window=5, min_periods=3).std()
    data['volatility_15d'] = data['returns'].rolling(window=15, min_periods=8).std()
    data['vol_ratio'] = data['volatility_5d'] / data['volatility_15d'].shift(1).replace(0, np.nan)
    
    data['vol_breakout'] = 0
    data.loc[data['vol_ratio'] > 2.0, 'vol_breakout'] = 1
    data.loc[data['vol_ratio'] < 0.5, 'vol_breakout'] = -1
    
    # Momentum-Velocity Divergence
    data['atr_3d'] = (
        pd.concat([
            data['high'] - data['low'],
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        ], axis=1).max(axis=1).rolling(window=3, min_periods=2).mean()
    )
    
    data['price_velocity'] = (data['close'] - data['close'].shift(2)) / data['atr_3d'].replace(0, np.nan)
    data['momentum_divergence'] = abs(data['price_velocity'] - data['regime_momentum'])
    
    # Signal Integration & Enhancement
    # Combine Momentum-Efficiency Components
    data['momentum_efficiency'] = data['regime_momentum'] * data['weighted_efficiency']
    data['momentum_efficiency_scaled'] = data['momentum_efficiency'] * data['volume_pressure_asymmetry']
    
    # Volume Confirmation
    data['avg_volume_10d'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_ratio'] = data['volume'] / data['avg_volume_10d']
    
    data['volume_multiplier'] = 1.0
    data.loc[data['volume_ratio'] > 2.0, 'volume_multiplier'] = 1.5
    data.loc[data['volume_ratio'] > 1.5, 'volume_multiplier'] = 1.2
    
    data['volume_confirmed'] = (
        data['momentum_efficiency_scaled'] * data['volume_multiplier'] * data['range_alignment']
    )
    
    # Enhance with Transition Signals
    data['transition_score'] = (
        0.6 * data['vol_breakout'] + 
        0.4 * np.sign(data['momentum_divergence'] - data['momentum_divergence'].rolling(window=10, min_periods=5).median())
    )
    
    data['transition_enhanced'] = data['volume_confirmed'] * (1 + data['transition_score'])
    
    # Dynamic Smoothing & Persistence
    smoothing_windows = {
        'high': 3,
        'low': 8,
        'normal': 5
    }
    
    data['smoothed_signal'] = np.nan
    for regime, window in smoothing_windows.items():
        mask = data['vol_regime'] == regime
        data.loc[mask, 'smoothed_signal'] = (
            data.loc[mask, 'transition_enhanced'].rolling(window=window, min_periods=window//2).mean()
        )
    
    # Momentum Persistence Filter
    data['signal_sign'] = np.sign(data['smoothed_signal'])
    data['sign_persistence'] = (
        data['signal_sign'].rolling(window=3, min_periods=3).apply(
            lambda x: 1 if len(set(x)) == 1 else 0, raw=False
        )
    )
    
    data['persistence_multiplier'] = 1.0
    data.loc[data['sign_persistence'] == 1, 'persistence_multiplier'] = 1.5
    
    # Final factor
    data['factor'] = data['smoothed_signal'] * data['persistence_multiplier']
    
    return data['factor']
