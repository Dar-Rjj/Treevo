import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap-Fractal Component
    # Gap Magnitude Scaling
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(5) - data['close'].shift(10))
    data['gap_magnitude'] = data['gap_magnitude'].replace([np.inf, -np.inf], np.nan)
    
    # Gap Fill Efficiency
    data['gap_fill_efficiency'] = (data['close'] - data['open']) / (data['close'].shift(1) - data['open'])
    data['gap_fill_efficiency'] = data['gap_fill_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Fractal Gap Score
    data['fractal_gap_score'] = data['gap_magnitude'] * data['gap_fill_efficiency']
    
    # Breakout-Fractal Patterns
    # Up Breakout Persistence
    up_breakout_condition = (data['high'] > data['high'].shift(1)) & (data['gap_magnitude'] > 0)
    data['up_breakout_persistence'] = up_breakout_condition.astype(int)
    for i in range(1, len(data)):
        if up_breakout_condition.iloc[i] and data['up_breakout_persistence'].iloc[i-1] > 0:
            data.iloc[i, data.columns.get_loc('up_breakout_persistence')] = data['up_breakout_persistence'].iloc[i-1] + 1
    
    # Down Breakout Persistence
    down_breakout_condition = (data['low'] < data['low'].shift(1)) & (data['gap_magnitude'] < 0)
    data['down_breakout_persistence'] = down_breakout_condition.astype(int)
    for i in range(1, len(data)):
        if down_breakout_condition.iloc[i] and data['down_breakout_persistence'].iloc[i-1] > 0:
            data.iloc[i, data.columns.get_loc('down_breakout_persistence')] = data['down_breakout_persistence'].iloc[i-1] + 1
    
    # Fractal Breakout Strength
    data['breakout_persistence'] = data['up_breakout_persistence'] - data['down_breakout_persistence']
    data['fractal_breakout_strength'] = data['breakout_persistence'] * data['fractal_gap_score']
    
    # Microstructure-Fractal Component
    # Range Efficiency Scaling
    data['range_efficiency_scaling'] = abs(data['close'] - data['open']) / (data['high'].shift(5) - data['low'].shift(5))
    data['range_efficiency_scaling'] = data['range_efficiency_scaling'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Clustering
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_clustering'] = data['volume'] / data['volume_5d_avg']
    
    # Fractal Efficiency
    data['fractal_efficiency'] = data['range_efficiency_scaling'] * data['volume_clustering']
    
    # Price-Volume Fractal Alignment
    # Buy-Sell Pressure Ratio
    data['buy_sell_pressure'] = (data['high'] - data['open']) / (data['open'] - data['low'])
    data['buy_sell_pressure'] = data['buy_sell_pressure'].replace([np.inf, -np.inf], np.nan)
    
    # Pressure-Fractal Coupling
    data['pressure_fractal_coupling'] = data['buy_sell_pressure'] * data['fractal_gap_score']
    
    # Microstructure Alignment
    data['microstructure_alignment'] = data['pressure_fractal_coupling'] * (1 - data['range_efficiency_scaling'])
    
    # Multi-Timeframe Regime Detection
    # Session Gap Analysis (using intraday proxies)
    # Early Gap Strength proxy: (High - Open) / |Open - Close_{t-1}|
    data['early_gap_strength'] = (data['high'] - data['open']) / abs(data['open'] - data['close'].shift(1))
    data['early_gap_strength'] = data['early_gap_strength'].replace([np.inf, -np.inf], np.nan)
    
    # Late Gap Filling proxy: (Close - Low) / |Open - Close_{t-1}|
    data['late_gap_filling'] = (data['close'] - data['low']) / abs(data['open'] - data['close'].shift(1))
    data['late_gap_filling'] = data['late_gap_filling'].replace([np.inf, -np.inf], np.nan)
    
    # Session Gap Divergence
    data['session_gap_divergence'] = data['early_gap_strength'] - data['late_gap_filling']
    
    # Volume-Pressure Regime
    # Volume Pressure
    data['volume_pressure'] = data['volume'] * data['buy_sell_pressure']
    
    # Pressure Regime Persistence
    pressure_direction = np.sign(data['volume_pressure'])
    data['pressure_regime_persistence'] = 1
    for i in range(1, len(data)):
        if pressure_direction.iloc[i] == pressure_direction.iloc[i-1] and pressure_direction.iloc[i] != 0:
            data.iloc[i, data.columns.get_loc('pressure_regime_persistence')] = data['pressure_regime_persistence'].iloc[i-1] + 1
    
    # Regime Strength
    data['regime_strength'] = data['volume_pressure'] * data['pressure_regime_persistence']
    
    # Composite Alpha Factor
    # Gap-Fractal Momentum
    data['gap_fractal_momentum'] = data['fractal_breakout_strength'] * data['session_gap_divergence']
    
    # Microstructure Confirmation
    data['microstructure_confirmation'] = data['microstructure_alignment'] * data['regime_strength']
    
    # Final Alpha
    data['alpha_factor'] = data['gap_fractal_momentum'] * data['microstructure_confirmation'] * data['fractal_efficiency']
    
    # Clean up intermediate columns and return final factor
    result = data['alpha_factor'].copy()
    
    return result
