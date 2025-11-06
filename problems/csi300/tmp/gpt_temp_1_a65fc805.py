import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Momentum Framework
    # Multi-Scale Momentum Divergence
    data['price_momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['momentum_3d'] = data['price_momentum'].rolling(window=3, min_periods=1).mean()
    data['momentum_8d'] = data['price_momentum'].rolling(window=8, min_periods=1).mean()
    data['momentum_divergence'] = data['momentum_3d'] - data['momentum_8d']
    
    # Volatility-Adjusted Momentum
    data['daily_volatility'] = data['high'] - data['low']
    data['vol_adj_momentum'] = data['momentum_divergence'] / (data['daily_volatility'] + 1e-8)
    data['vol_momentum'] = data['daily_volatility'] / data['daily_volatility'].rolling(window=5, min_periods=1).mean() - 1
    
    # Fractal Momentum Patterns
    data['short_term_fractal'] = data['momentum_divergence'] / (data['momentum_divergence'].rolling(window=3, min_periods=1).mean() + 1e-8)
    data['medium_term_fractal'] = data['momentum_divergence'] / (data['momentum_divergence'].rolling(window=8, min_periods=1).mean() + 1e-8)
    data['fractal_convergence'] = data['short_term_fractal'] - data['medium_term_fractal']
    
    # Efficiency Spectrum Analysis
    # Multi-Dimensional Efficiency
    data['open_close_efficiency'] = abs(data['close'] - data['open']) / (data['daily_volatility'] + 1e-8)
    data['high_low_efficiency'] = data['daily_volatility'] / (data['high'].rolling(window=5, min_periods=1).max() - data['low'].rolling(window=5, min_periods=1).min() + 1e-8)
    data['gap_efficiency'] = abs(data['open'] - data['close'].shift(1)) / (data['daily_volatility'] + 1e-8)
    
    # Efficiency Momentum Dynamics
    data['open_close_eff_momentum'] = data['open_close_efficiency'] / data['open_close_efficiency'].rolling(window=5, min_periods=1).mean() - 1
    data['high_low_eff_momentum'] = data['high_low_efficiency'] / data['high_low_efficiency'].rolling(window=5, min_periods=1).mean() - 1
    data['gap_eff_momentum'] = data['gap_efficiency'] / data['gap_efficiency'].rolling(window=5, min_periods=1).mean() - 1
    
    # Efficiency Divergence Patterns
    data['efficiency_spread'] = data['open_close_eff_momentum'] - data['high_low_eff_momentum']
    data['gap_efficiency_alignment'] = data['gap_eff_momentum'] - data['open_close_eff_momentum']
    data['efficiency_convergence'] = data['efficiency_spread'].rolling(window=3, min_periods=1).mean() - data['efficiency_spread'].rolling(window=8, min_periods=1).mean()
    
    # Volume-Pressure Integration
    # Volume Flow Analysis
    data['volume_pressure'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_acceleration'] = data['volume_pressure'] / data['volume_pressure'].rolling(window=3, min_periods=1).mean() - 1
    
    def count_above_one(series):
        return series.rolling(window=5, min_periods=1).apply(lambda x: (x > 1).sum(), raw=True)
    
    data['volume_persistence'] = count_above_one(data['volume_pressure'])
    
    # Amount-Based Pressure
    data['amount_intensity'] = data['amount'] / data['amount'].rolling(window=5, min_periods=1).mean()
    data['amount_volume_ratio'] = data['amount_intensity'] / (data['volume_pressure'] + 1e-8)
    data['large_trade_detection'] = (data['amount_volume_ratio'] > data['amount_volume_ratio'].rolling(window=10, min_periods=1).mean()).astype(int)
    
    # Pressure-Momentum Alignment
    data['volume_confirmed_momentum'] = data['momentum_divergence'] * data['volume_acceleration']
    data['amount_weighted_efficiency'] = data['efficiency_spread'] * data['amount_intensity']
    data['pressure_divergence'] = data['volume_acceleration'] - data['amount_intensity']
    
    # Fractal Regime Detection
    # Multi-Timeframe Volatility Structure
    data['short_term_vol'] = data['daily_volatility'].rolling(window=3, min_periods=1).mean()
    data['medium_term_vol'] = data['daily_volatility'].rolling(window=8, min_periods=1).mean()
    data['volatility_ratio'] = data['short_term_vol'] / (data['medium_term_vol'] + 1e-8)
    data['volatility_regime'] = np.where(data['volatility_ratio'] > 1.2, 1, 0)  # 1: expanding, 0: contracting
    
    # Efficiency Regime Analysis
    data['efficiency_volatility'] = data['open_close_efficiency'].rolling(window=5, min_periods=1).std()
    data['efficiency_regime'] = np.where(data['efficiency_volatility'] > data['efficiency_volatility'].rolling(window=10, min_periods=1).mean(), 1, 0)
    
    def count_same_regime(series):
        return series.rolling(window=3, min_periods=1).apply(lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0, raw=False)
    
    data['regime_persistence'] = count_same_regime(data['efficiency_regime'])
    
    # Regime-Adaptive Signals
    data['volatility_scaled_fractal'] = data['fractal_convergence'] * data['volatility_ratio']
    data['breakout_momentum'] = data['momentum_divergence'] * (data['volatility_ratio'] - 1)
    
    def count_below_mean(series):
        return series.rolling(window=5, min_periods=1).apply(lambda x: (x < x.mean()).sum(), raw=True)
    
    data['compression_detection'] = count_below_mean(data['daily_volatility'])
    data['efficiency_breakout'] = data['efficiency_spread'] * (1 / (data['volatility_ratio'] + 1e-8))
    
    data['variance_weighted_momentum'] = data['momentum_divergence'] * data['efficiency_volatility']
    data['regime_confirmed_efficiency'] = data['efficiency_convergence'] * data['regime_persistence']
    
    # Intraday Structure Confirmation
    # Session-Based Momentum
    data['morning_momentum'] = (data['high'] - data['open']) / (data['daily_volatility'] + 1e-8)
    data['afternoon_momentum'] = (data['close'] - data['low']) / (data['daily_volatility'] + 1e-8)
    data['session_momentum_divergence'] = data['morning_momentum'] - data['afternoon_momentum']
    
    # Price-Level Analysis
    data['relative_position'] = (data['close'] - data['low']) / (data['daily_volatility'] + 1e-8)
    data['position_momentum'] = data['relative_position'] / data['relative_position'].rolling(window=5, min_periods=1).mean() - 1
    data['extreme_position'] = ((data['relative_position'] > 0.8) | (data['relative_position'] < 0.2)).astype(int)
    
    # Structure-Momentum Integration
    data['session_confirmed_momentum'] = data['momentum_divergence'] * data['session_momentum_divergence']
    data['position_weighted_efficiency'] = data['efficiency_spread'] * data['position_momentum']
    data['structure_alignment'] = data['session_momentum_divergence'] * data['position_momentum']
    
    # Composite Alpha Construction
    # Core components
    core_momentum = data['vol_adj_momentum'] + data['fractal_convergence'] + data['momentum_divergence']
    
    # Efficiency enhancement
    efficiency_enhancement = (data['efficiency_spread'] + data['efficiency_convergence'] + 
                             data['open_close_eff_momentum'] + data['high_low_eff_momentum'] + data['gap_eff_momentum'])
    
    # Volume-pressure confirmation
    volume_confirmation = (data['volume_confirmed_momentum'] + data['amount_weighted_efficiency'] + 
                          data['pressure_divergence'])
    
    # Fractal regime adaptation
    regime_weights = (data['volatility_scaled_fractal'] + data['breakout_momentum'] + 
                     data['efficiency_breakout'] + data['variance_weighted_momentum'] + 
                     data['regime_confirmed_efficiency'])
    
    # Intraday structure filtering
    intraday_signals = (data['session_confirmed_momentum'] + data['position_weighted_efficiency'] + 
                       data['structure_alignment'])
    
    # Final signal synthesis
    alpha_signal = (core_momentum * 0.3 + 
                   efficiency_enhancement * 0.25 + 
                   volume_confirmation * 0.2 + 
                   regime_weights * 0.15 + 
                   intraday_signals * 0.1)
    
    return alpha_signal
