import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Gap-Momentum Framework
    data['intraday_gap_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_gap_efficiency'] = data['intraday_gap_efficiency'].replace([np.inf, -np.inf], 0)
    
    data['overnight_gap_momentum'] = (data['open'] - data['close'].shift(1)) / np.abs(data['open'] - data['close'].shift(1))
    data['overnight_gap_momentum'] = data['overnight_gap_momentum'].fillna(0)
    
    # Medium-Term Gap Efficiency
    data['medium_term_gap_efficiency'] = np.abs(data['close'] - data['open'].shift(5)) / (
        (data['high'] - data['low']).rolling(window=6, min_periods=1).sum()
    )
    data['medium_term_gap_efficiency'] = data['medium_term_gap_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['gap_efficiency_divergence'] = data['intraday_gap_efficiency'] - data['medium_term_gap_efficiency']
    
    # Fractal Volatility Construction
    data['micro_volatility'] = (data['high'] - data['low']) / data['close']
    data['micro_volatility'] = data['micro_volatility'].replace([np.inf, -np.inf], 0)
    
    data['meso_volatility'] = (
        data['high'].rolling(window=3, min_periods=1).max() - 
        data['low'].rolling(window=3, min_periods=1).min()
    ) / data['close'].rolling(window=3, min_periods=1).mean()
    data['meso_volatility'] = data['meso_volatility'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['macro_volatility'] = (
        data['high'].rolling(window=5, min_periods=1).max() - 
        data['low'].rolling(window=5, min_periods=1).min()
    ) / data['close'].rolling(window=5, min_periods=1).mean()
    data['macro_volatility'] = data['macro_volatility'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume-Pressure Confirmation System
    data['morning_gap_pressure'] = (data['high'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['morning_gap_pressure'] = data['morning_gap_pressure'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['gap_fill_pressure'] = (data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['gap_fill_pressure'] = data['gap_fill_pressure'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['pressure_asymmetry'] = data['morning_gap_pressure'] - data['gap_fill_pressure']
    
    # Volume Asymmetry Analysis
    returns = data['close'].pct_change()
    up_days = returns > 0
    data['upside_volume_ratio'] = (
        data['volume'].rolling(window=10, min_periods=1).apply(
            lambda x: x[up_days.loc[x.index]].mean() if up_days.loc[x.index].any() else 0
        ) / data['volume'].rolling(window=10, min_periods=1).mean()
    ).fillna(0)
    
    positive_returns_sum = returns.clip(lower=0).rolling(window=10, min_periods=1).sum()
    negative_returns_sum = (-returns).clip(lower=0).rolling(window=10, min_periods=1).sum()
    data['price_asymmetry'] = np.log1p(positive_returns_sum) - np.log1p(negative_returns_sum)
    data['price_asymmetry'] = data['price_asymmetry'].fillna(0)
    
    data['volume_asymmetry'] = data['upside_volume_ratio'] * data['price_asymmetry']
    
    data['volume_pressure_confirmation'] = np.cbrt(data['pressure_asymmetry'] * data['volume_asymmetry'])
    data['volume_pressure_confirmation'] = data['volume_pressure_confirmation'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Regime-Based Momentum Enhancement
    micro_vol_avg = data['micro_volatility'].rolling(window=20, min_periods=1).mean()
    data['volatility_regime'] = np.where(
        data['micro_volatility'] > 1.2 * micro_vol_avg, 'high',
        np.where(data['micro_volatility'] < 0.8 * micro_vol_avg, 'low', 'normal')
    )
    
    data['regime_gap_momentum'] = np.where(
        data['volatility_regime'] == 'high',
        data['intraday_gap_efficiency'] * data['gap_efficiency_divergence'] * data['micro_volatility'],
        np.where(
            data['volatility_regime'] == 'low',
            data['overnight_gap_momentum'] * data['volume_pressure_confirmation'] * data['macro_volatility'],
            (data['intraday_gap_efficiency'] + data['overnight_gap_momentum']) * data['meso_volatility']
        )
    )
    
    # Volume Cluster Dynamics
    data['volume_spike_detection'] = (data['volume'] > 1.5 * data['volume'].rolling(window=20, min_periods=1).mean()).astype(float)
    
    gap_volume = np.abs(data['open'] - data['close'].shift(1))
    max_gap_7 = gap_volume.rolling(window=7, min_periods=1).max()
    min_gap_7 = gap_volume.rolling(window=7, min_periods=1).min()
    max_gap_2 = gap_volume.rolling(window=2, min_periods=1).max()
    min_gap_2 = gap_volume.rolling(window=2, min_periods=1).min()
    
    data['gap_volume_fractal'] = np.log(np.maximum(max_gap_7 - min_gap_7, 1e-6)) / np.log(np.maximum(max_gap_2 - min_gap_2, 1e-6))
    data['gap_volume_fractal'] = data['gap_volume_fractal'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['volume_cluster_momentum'] = (data['volume_spike_detection'] - 0.5) * data['gap_volume_fractal']
    
    # Dynamic Alpha Synthesis
    data['regime_gap_momentum_signal'] = data['regime_gap_momentum'] * data['gap_efficiency_divergence']
    data['pressure_confirmation_signal'] = data['volume_pressure_confirmation'] * data['pressure_asymmetry']
    data['volume_cluster_signal'] = data['volume_cluster_momentum'] * data['volume_asymmetry']
    
    data['base_alpha'] = data['regime_gap_momentum_signal'] + data['pressure_confirmation_signal'] + data['volume_cluster_signal']
    
    # Signal Refinement & Risk Adjustment
    # True Range calculation
    tr1 = data['high'] - data['low']
    tr2 = np.abs(data['high'] - data['close'].shift(1))
    tr3 = np.abs(data['low'] - data['close'].shift(1))
    data['true_range'] = np.maximum(np.maximum(tr1, tr2), tr3)
    data['true_range_volatility'] = data['true_range'].rolling(window=20, min_periods=1).mean()
    
    data['high_low_range_expansion'] = (
        data['high'].rolling(window=20, min_periods=1).max() / 
        data['low'].rolling(window=20, min_periods=1).min() - 1
    )
    
    # Price-Volume Confirmation
    data['price_confirmation'] = (
        (data['close'] > data['close'].shift(1)).astype(float) *
        (data['close'] > data['close'].shift(3)).astype(float) *
        (data['close'] > data['close'].shift(5)).astype(float)
    )
    
    data['volume_confirmation'] = (
        (data['volume'] > data['volume'].shift(1)).astype(float) *
        (data['volume'] > data['volume'].rolling(window=5, min_periods=1).mean()).astype(float)
    )
    
    data['confirmation_strength'] = 1 + data['price_confirmation'] * data['volume_confirmation']
    
    # Final Alpha
    data['final_alpha'] = (
        data['base_alpha'] * 
        data['confirmation_strength'] / 
        np.maximum(data['true_range_volatility'], 1e-6) * 
        data['high_low_range_expansion']
    )
    data['final_alpha'] = data['final_alpha'].replace([np.inf, -np.inf], 0).fillna(0)
    
    return data['final_alpha']
