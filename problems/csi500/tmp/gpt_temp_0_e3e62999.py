import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility Asymmetry with Price-Volume Fractal Dynamics
    """
    data = df.copy()
    
    # 1. Asymmetric Volatility Structure Analysis
    # Directional Volatility Decomposition
    data['max_open_close'] = data[['open', 'close']].max(axis=1)
    data['min_open_close'] = data[['open', 'close']].min(axis=1)
    
    # Upward and Downward Volatility Components
    data['upward_vol'] = data['high'] - data['max_open_close']
    data['downward_vol'] = data['min_open_close'] - data['low']
    data['total_range'] = data['high'] - data['low']
    
    # Volatility Ratios
    data['upward_ratio'] = np.where(data['total_range'] > 0, 
                                   data['upward_vol'] / data['total_range'], 0)
    data['downward_ratio'] = np.where(data['total_range'] > 0, 
                                     data['downward_vol'] / data['total_range'], 0)
    
    # Volatility Asymmetry Index
    data['asymmetry'] = data['upward_ratio'] - data['downward_ratio']
    data['asymmetry_persistence'] = data['asymmetry'] - data['asymmetry'].shift(1)
    
    # Multi-Timeframe Volatility
    data['intraday_vol'] = (data['high'] - data['low']) / data['open']
    data['short_term_vol'] = data['intraday_vol'].rolling(window=3, min_periods=1).mean()
    data['vol_mean_reversion'] = data['intraday_vol'] - data['short_term_vol']
    
    # Volatility Regime Classification
    vol_std = data['intraday_vol'].rolling(window=5, min_periods=3).std()
    vol_trend = data['intraday_vol'].rolling(window=3, min_periods=2).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.mean() if x.mean() > 0 else 0
    )
    
    # 2. Fractal Price-Volume Dynamics
    # Multi-Scale Volume Fractals
    data['volume_change'] = data['volume'].pct_change()
    data['volume_persistence'] = np.sign(data['volume_change']) * np.sign(data['volume_change'].shift(1))
    
    # Short-term volume variance (3-day)
    data['volume_var_short'] = data['volume'].rolling(window=3, min_periods=2).std()
    data['volume_var_medium'] = data['volume'].rolling(window=5, min_periods=3).std()
    
    # Price-Volume Fractal Alignment
    data['price_change'] = data['close'].pct_change()
    data['up_move_volume_eff'] = np.where(data['close'] > data['open'],
                                         (data['close'] - data['open']) / data['volume'], 0)
    data['down_move_volume_eff'] = np.where(data['close'] < data['open'],
                                           (data['open'] - data['close']) / data['volume'], 0)
    
    # Fractal Momentum Indicators
    data['momentum_short'] = data['close'].pct_change(periods=2)
    data['momentum_medium'] = data['close'].pct_change(periods=5)
    data['momentum_fractal_ratio'] = np.where(
        np.abs(data['momentum_medium']) > 0,
        np.abs(data['momentum_short']) / np.abs(data['momentum_medium']), 1
    )
    
    # 3. Asymmetric Response Framework
    # Volatility Asymmetry Response
    data['up_vol_sensitivity'] = data['upward_ratio'] * data['up_move_volume_eff']
    data['down_vol_sensitivity'] = data['downward_ratio'] * data['down_move_volume_eff']
    
    # Fractal Dynamics Response
    data['fractal_convergence'] = np.where(
        np.sign(data['momentum_short']) == np.sign(data['momentum_medium']),
        np.abs(data['momentum_short'] + data['momentum_medium']), 0
    )
    
    # 4. Predictive Factor Construction
    # Core Asymmetry-Fractal Components
    data['vol_asymmetry_momentum'] = data['asymmetry'] * data['momentum_fractal_ratio']
    data['asymmetry_persistence_volume'] = data['asymmetry_persistence'] * data['volume_var_short']
    
    # Regime-Adaptive Weighting
    # Volatility Regime Classification
    expanding_vol = (vol_trend > 0.1) & (vol_std > vol_std.rolling(window=10, min_periods=5).mean())
    contracting_vol = (vol_trend < -0.1) & (vol_std < vol_std.rolling(window=10, min_periods=5).mean())
    stable_vol = (~expanding_vol) & (~contracting_vol)
    
    # Fractal Strength
    strong_fractal = (data['fractal_convergence'] > data['fractal_convergence'].rolling(window=10, min_periods=5).quantile(0.7))
    weak_fractal = (data['fractal_convergence'] < data['fractal_convergence'].rolling(window=10, min_periods=5).quantile(0.3))
    
    # Volume Confirmation
    high_volume_conc = data['volume_var_short'] > data['volume_var_short'].rolling(window=10, min_periods=5).quantile(0.7)
    
    # Apply regime weights
    volatility_multiplier = np.where(expanding_vol, 1.4, 
                                   np.where(contracting_vol, 1.2, 0.8))
    
    fractal_multiplier = np.where(strong_fractal, 1.3,
                                np.where(weak_fractal, 0.7, 1.1))
    
    volume_multiplier = np.where(high_volume_conc, 1.25, 1.0)
    
    # Final factor construction
    core_factor = (
        data['vol_asymmetry_momentum'] * 0.4 +
        data['asymmetry_persistence_volume'] * 0.3 +
        data['fractal_convergence'] * 0.2 +
        (data['up_vol_sensitivity'] - data['down_vol_sensitivity']) * 0.1
    )
    
    # Apply regime-adaptive weighting
    final_factor = core_factor * volatility_multiplier * fractal_multiplier * volume_multiplier
    
    # Normalize the factor
    factor_series = (final_factor - final_factor.rolling(window=20, min_periods=10).mean()) / final_factor.rolling(window=20, min_periods=10).std()
    
    return factor_series
