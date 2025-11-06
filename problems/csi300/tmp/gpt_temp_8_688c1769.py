import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Period Efficiency Assessment
    data['daily_range'] = data['high'] - data['low']
    data['prev_close'] = data['close'].shift(1)
    data['range_efficiency'] = np.abs(data['close'] - data['prev_close']) / data['daily_range']
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Directional efficiency components
    up_mask = data['close'] > data['prev_close']
    down_mask = data['close'] < data['prev_close']
    data['up_efficiency'] = np.where(up_mask, (data['close'] - data['prev_close']) / data['daily_range'], 0)
    data['down_efficiency'] = np.where(down_mask, (data['prev_close'] - data['close']) / data['daily_range'], 0)
    
    # Efficiency momentum
    data['efficiency_momentum'] = data['range_efficiency'] / data['range_efficiency'].rolling(window=5, min_periods=1).mean()
    
    # Fractal Volume-Range Microstructure
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_clustering'] = data['volume_ratio'].rolling(window=5, min_periods=1).std()
    
    # Volume fractal dimension (simplified Hurst)
    def hurst_approx(series, window=10):
        lags = range(2, min(6, window))
        tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    data['volume_fractal'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: hurst_approx(pd.Series(x)) if len(x) >= 5 else np.nan, raw=False
    )
    
    # Range efficiency & position ratio
    data['position_ratio'] = (data['close'] - data['low']) / data['daily_range']
    data['position_ratio'] = data['position_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Fractal Efficiency Ratio
    data['fractal_efficiency_ratio'] = np.abs(data['close'] - data['prev_close']) / (data['volume'] ** 0.5)
    
    # Microstructural Regime Detection
    data['range_10d_avg'] = data['daily_range'].rolling(window=10, min_periods=1).mean()
    data['range_breakout'] = data['daily_range'] / data['range_10d_avg']
    
    # Volume-Range Synchronization
    data['volume_range_corr'] = data['volume'].rolling(window=10, min_periods=5).corr(data['daily_range'])
    
    # Volume-Weighted Reversal System
    data['short_term_reversal'] = (data['close'].shift(1) - data['close'].shift(3)) / data['close'].shift(3)
    data['medium_term_reversal'] = (data['close'].shift(1) - data['close'].shift(6)) / data['close'].shift(6)
    
    # Reversal acceleration
    data['reversal_acceleration'] = (data['short_term_reversal'] - data['medium_term_reversal']) / (
        np.abs(data['medium_term_reversal']) + 1e-8)
    
    # Volume Momentum Analysis
    data['volume_4d_mean'] = data['volume'].rolling(window=4, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume_4d_mean'] - 1
    
    data['volume_9d_mean'] = data['volume'].rolling(window=9, min_periods=1).mean()
    data['volume_trend'] = data['volume_4d_mean'] / data['volume_9d_mean'] - 1
    
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Volume-Confirmed Reversal
    data['volume_confirmed_reversal'] = data['reversal_acceleration'] * data['volume_ratio'] * np.sign(data['volume_trend'])
    
    # Efficiency-Volume Momentum
    data['efficiency_volume_momentum'] = (data['up_efficiency'] - data['down_efficiency']) * data['volume_acceleration']
    data['range_volume_momentum'] = data['range_efficiency'] * data['volume_trend']
    
    # Volatility-Fractal Framework
    # True Range Calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    
    # Volatility components
    data['opening_gap_vol'] = np.abs(data['open'] - data['prev_close']) / (data['atr_5'] + 1e-8)
    data['core_session_vol'] = data['daily_range'] / (data['atr_5'] + 1e-8)
    data['closing_vol'] = np.abs(data['close'] - (data['high'] + data['low'])/2) / (data['atr_5'] + 1e-8)
    
    # Volatility regime
    data['atr_3'] = data['true_range'].rolling(window=3, min_periods=1).mean()
    data['atr_8'] = data['true_range'].rolling(window=8, min_periods=1).mean()
    data['volatility_regime'] = data['atr_3'] / data['atr_8']
    
    data['volatility_ratio'] = data['true_range'] / (data['atr_5'] + 1e-8)
    data['volatility_trend'] = data['atr_3'] / data['atr_8']
    
    # Adaptive Signal Integration
    # Efficiency Volatility-Fractal Adjustment
    data['scaled_efficiency'] = (data['up_efficiency'] - data['down_efficiency']) * data['volatility_ratio']
    data['low_vol_efficiency'] = np.where(
        data['volatility_regime'] < 1, 
        (data['up_efficiency'] - data['down_efficiency']) * data['volume_fractal'],
        0
    )
    data['vol_weighted_efficiency_momentum'] = data['efficiency_momentum'] * data['volatility_trend']
    
    # Reversal Volatility-Fractal Context
    data['scaled_reversal'] = data['reversal_acceleration'] / (data['atr_3'] + 1e-8)
    data['vol_fractal_reversal'] = data['scaled_reversal'] * data['volatility_trend'] * data['volume_range_corr']
    data['volume_cluster_reversal'] = data['reversal_acceleration'] * data['volume_clustering']
    
    # Volume-Range Volatility-Fractal Scaling
    data['scaled_volume_range'] = data['volume_confirmed_reversal'] * data['volatility_regime']
    data['fractal_efficiency_confirmation'] = data['fractal_efficiency_ratio'] * data['volume_confirmed_reversal']
    
    # Composite Signal Construction
    # Core Factor Integration
    data['vol_fractal_efficiency'] = (
        (data['scaled_efficiency'] + data['low_vol_efficiency']) * 
        data['volatility_regime'] * data['volume_range_corr']
    )
    
    data['volume_weighted_reversal'] = (
        data['efficiency_volume_momentum'] + 
        data['volume_confirmed_reversal']
    ) * data['volatility_regime'] * data['range_efficiency'] * data['volume_fractal']
    
    # Cross-Validation Framework
    data['efficiency_volume_alignment'] = (
        np.sign(data['scaled_efficiency']) * np.sign(data['volume_acceleration']) * 
        np.sign(data['volume_fractal'] - 0.5)
    )
    
    data['reversal_range_convergence'] = (
        np.sign(data['reversal_acceleration']) * np.sign(1 - data['range_breakout']) * 
        np.sign(data['volume_range_corr'])
    )
    
    # Final Signal Synthesis
    alpha_factor = (
        data['vol_fractal_efficiency'] * 0.4 +
        data['volume_weighted_reversal'] * 0.3 +
        data['efficiency_volume_alignment'] * data['vol_fractal_efficiency'] * 0.15 +
        data['reversal_range_convergence'] * data['volume_weighted_reversal'] * 0.15
    )
    
    return alpha_factor
