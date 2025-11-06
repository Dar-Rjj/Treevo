import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Market State Classification
    data['volatility_regime'] = (data['high'] - data['low']) / data['close']
    data['volume_median'] = data['volume'].rolling(window=10, min_periods=1).median()
    data['volume_activity'] = data['volume'] > (2 * data['volume_median'])
    data['market_state'] = np.where((data['volatility_regime'] > data['volatility_regime'].rolling(window=20, min_periods=1).median()) & 
                                   data['volume_activity'], 'Active', 'Stable')
    
    # Fractal Entropy Dynamics
    # Price-Volume Entropy
    data['price_avg_5'] = data['close'].rolling(window=5, min_periods=1).mean()
    data['price_dev_sq'] = (data['close'] - data['price_avg_5']) ** 2
    data['price_volume_entropy'] = -(data['price_dev_sq'] * data['volume']) / data['volume'].rolling(window=5, min_periods=1).sum()
    
    # Volume Fractal Dimension
    data['vol_range'] = data['high'] - data['low']
    data['vol_range_prev'] = data['vol_range'].shift(1)
    data['volume_prev'] = data['volume'].shift(1)
    data['volume_fractal_dim'] = np.log(data['volume'] / data['volume_prev']) / np.log(data['vol_range'] / data['vol_range_prev'])
    data['volume_fractal_dim'] = data['volume_fractal_dim'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volatility Efficiency
    data['volatility_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['volatility_efficiency'] = data['volatility_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Momentum-Entropy Integration
    data['short_term_momentum'] = ((data['close'] - data['close'].shift(3)) / data['close'].shift(3)) * (1 - data['price_volume_entropy'])
    data['medium_term_momentum'] = (data['close'] / data['close'].shift(5) - 1) * data['volume_fractal_dim']
    data['momentum_entropy_divergence'] = (data['short_term_momentum'] - data['medium_term_momentum']) * data['volume_fractal_dim']
    
    # Entropy-Weighted Efficiency
    data['efficiency_momentum_alignment'] = data['volatility_efficiency'] * data['short_term_momentum']
    data['fractal_momentum_quality'] = data['volume_fractal_dim'] * data['medium_term_momentum']
    data['entropy_regime_detection'] = abs(data['short_term_momentum'] / data['medium_term_momentum'] - 1) * data['volatility_efficiency']
    data['entropy_regime_detection'] = data['entropy_regime_detection'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Pressure-Momentum Patterns
    # Order Flow Pressure Analysis
    data['buy_side_pressure'] = (data['high'] - data['open']) * data['volume']
    data['sell_side_pressure'] = (data['open'] - data['low']) * data['volume']
    data['pressure_asymmetry_ratio'] = data['buy_side_pressure'] / data['sell_side_pressure']
    data['pressure_asymmetry_ratio'] = data['pressure_asymmetry_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Volume-Efficiency Integration
    # Up-Tick and Down-Tick Volume
    data['close_prev'] = data['close'].shift(1)
    data['up_tick'] = data['close'] > data['close_prev']
    data['down_tick'] = data['close'] < data['close_prev']
    
    up_volume = []
    down_volume = []
    for i in range(len(data)):
        if i < 5:
            up_volume.append(data['volume'].iloc[:i+1][data['up_tick'].iloc[:i+1]].sum())
            down_volume.append(data['volume'].iloc[:i+1][data['down_tick'].iloc[:i+1]].sum())
        else:
            up_volume.append(data['volume'].iloc[i-4:i+1][data['up_tick'].iloc[i-4:i+1]].sum())
            down_volume.append(data['volume'].iloc[i-4:i+1][data['down_tick'].iloc[i-4:i+1]].sum())
    
    data['up_tick_volume'] = up_volume
    data['down_tick_volume'] = down_volume
    data['sustained_volume_ratio'] = data['up_tick_volume'] / data['down_tick_volume']
    data['sustained_volume_ratio'] = data['sustained_volume_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Pressure-Momentum Coupling
    data['efficiency_pressure_alignment'] = data['volatility_efficiency'] * data['pressure_asymmetry_ratio']
    data['volume_momentum_divergence'] = data['volume_fractal_dim'] * data['momentum_entropy_divergence']
    data['nonlinear_pressure_regime'] = abs(data['volatility_efficiency'] - data['pressure_asymmetry_ratio']) * data['volume_fractal_dim']
    
    # Entropy Compression Breakout System
    # Range Compression Identification
    data['high_avg_5'] = data['high'].rolling(window=5, min_periods=1).mean()
    data['low_avg_5'] = data['low'].rolling(window=5, min_periods=1).mean()
    data['range_stability'] = (data['high'] - data['low']) / (data['high_avg_5'] - data['low_avg_5'])
    data['range_stability'] = data['range_stability'].replace([np.inf, -np.inf], np.nan).fillna(1)
    data['compression_threshold'] = data['range_stability'] < 0.7
    data['entropy_compression'] = data['volume_fractal_dim'] < 0.5
    
    # Momentum Breakout Patterns
    data['high_max_5'] = data['high'].rolling(window=5, min_periods=1).max()
    data['price_breakout'] = data['close'] > data['high_max_5'].shift(1)
    data['volume_confirmation'] = data['volume'] > (1.5 * data['volume'].shift(1))
    data['momentum_breakout_quality'] = ((data['high'] - data['close'].shift(1)) / (data['close'].shift(1) - data['low'])) * data['short_term_momentum']
    data['momentum_breakout_quality'] = data['momentum_breakout_quality'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Entropy-Enhanced Breakout
    data['entropy_breakout_alignment'] = data['price_breakout'].astype(float) * data['momentum_entropy_divergence']
    data['volume_efficiency_confirmation'] = data['volume_confirmation'].astype(float) * data['volatility_efficiency']
    data['pressure_momentum_synchronization'] = data['pressure_asymmetry_ratio'] * data['momentum_breakout_quality']
    
    # Composite Signal Generation
    # Primary Signal Components
    data['momentum_pressure_factor'] = data['fractal_momentum_quality'] * data['efficiency_pressure_alignment']
    data['entropy_breakout_efficiency'] = data['momentum_breakout_quality'] * data['volatility_efficiency']
    data['regime_adaptive_weighting'] = data['nonlinear_pressure_regime'] * data['volume_fractal_dim']
    
    # Divergence Enhancement
    data['multi_timeframe_alignment'] = data['momentum_entropy_divergence'] * data['volume_momentum_divergence']
    data['pressure_volume_convergence'] = np.sign(data['pressure_asymmetry_ratio'] - 1) * np.sign(data['volume_fractal_dim'] - 1)
    data['entropy_breakout_filter'] = data['entropy_compression'].astype(float) * data['entropy_breakout_alignment']
    
    # Cross-Validation Components
    data['amount_avg_5'] = data['amount'].rolling(window=5, min_periods=1).mean()
    data['amount_activity'] = data['amount'] / data['amount_avg_5']
    data['sustained_momentum_confirmation'] = data['sustained_volume_ratio'] * data['volume_momentum_divergence']
    data['volatility_regime_filter'] = data['entropy_regime_detection'] * data['range_stability']
    
    # Final Alpha Factor Synthesis
    data['core_factor'] = data['momentum_pressure_factor'] * data['entropy_breakout_efficiency']
    
    # State-Adaptive Enhancement
    active_mask = data['market_state'] == 'Active'
    stable_mask = data['market_state'] == 'Stable'
    
    data['state_enhanced_factor'] = 0
    data.loc[active_mask, 'state_enhanced_factor'] = (data['core_factor'] * data['volatility_efficiency'] * 
                                                     data['entropy_regime_detection'])
    data.loc[stable_mask, 'state_enhanced_factor'] = (data['core_factor'] * data['sustained_volume_ratio'] * 
                                                     data['volume_fractal_dim'])
    
    # Divergence Signal
    data['divergence_signal'] = data['core_factor'] * data['multi_timeframe_alignment']
    
    # Final Fractal Momentum-Entropy Pressure Factor
    data['fractal_momentum_entropy_pressure'] = (data['state_enhanced_factor'] + data['divergence_signal']) / 2
    
    # Clean up intermediate columns
    result = data['fractal_momentum_entropy_pressure']
    
    return result
