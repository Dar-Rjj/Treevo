import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Realized Volatility Estimation
    data['ret'] = data['close'].pct_change()
    data['short_term_vol'] = data['ret'].rolling(window=5).apply(lambda x: np.sqrt(np.sum(x**2)), raw=True)
    data['medium_term_vol'] = data['ret'].rolling(window=10).apply(lambda x: np.sqrt(np.sum(x**2)), raw=True)
    
    # Volatility Ratio and Regime Classification
    data['vol_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    data['vol_regime'] = 'normal'
    data.loc[data['vol_ratio'] > 1.2, 'vol_regime'] = 'high'
    data.loc[data['vol_ratio'] < 0.8, 'vol_regime'] = 'low'
    
    # Price-Volume Entropy Analysis
    def calculate_entropy(volume_series, condition_series):
        volume_sum = volume_series.sum()
        if volume_sum == 0:
            return 0
        weights = volume_series / volume_sum
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        return entropy
    
    # Directional Entropy
    data['up_entropy'] = 0.0
    data['down_entropy'] = 0.0
    
    for i in range(len(data)):
        if i >= 4:
            volume_window = data['volume'].iloc[i-4:i+1]
            close_window = data['close'].iloc[i-4:i+1]
            
            up_mask = close_window > close_window.shift(1)
            down_mask = close_window < close_window.shift(1)
            
            up_volume = volume_window.where(up_mask, 0)
            down_volume = volume_window.where(down_mask, 0)
            
            data.loc[data.index[i], 'up_entropy'] = calculate_entropy(up_volume, up_mask)
            data.loc[data.index[i], 'down_entropy'] = calculate_entropy(down_volume, down_mask)
    
    # Entropy Differential and Persistence
    data['directional_bias'] = (data['up_entropy'] - data['down_entropy']) / (data['up_entropy'] + data['down_entropy'] + 1e-10)
    data['entropy_momentum'] = data['directional_bias'] - data['directional_bias'].shift(3)
    data['entropy_acceleration'] = (data['directional_bias'] - data['directional_bias'].shift(3)) - (data['directional_bias'].shift(3) - data['directional_bias'].shift(6))
    
    # Fractal Microstructure Analysis
    data['micro_range'] = (data['high'] - data['low']) / data['close']
    data['meso_range'] = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / data['close'].rolling(window=3).mean()
    data['macro_range'] = (data['high'].rolling(window=8).max() - data['low'].rolling(window=8).min()) / data['close'].rolling(window=8).mean()
    
    # Range Complexity (Fractal Dimension)
    data['range_complexity'] = np.log(data['meso_range'] / (data['micro_range'] + 1e-10)) / np.log(data['macro_range'] / (data['meso_range'] + 1e-10))
    
    # Fractal Momentum
    data['micro_meso_momentum'] = (data['micro_range'] - data['micro_range'].shift(3)) / (data['micro_range'].shift(3) + 1e-10)
    data['meso_macro_momentum'] = (data['meso_range'] - data['meso_range'].shift(3)) / (data['meso_range'].shift(3) + 1e-10)
    
    # Opening-Closing Dynamics
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-10)
    data['closing_pressure'] = (data['close'] - data['open']) / (data['open'] + 1e-10)
    
    data['opening_efficiency'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-10)
    data['closing_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-10)
    
    data['pressure_asymmetry'] = data['opening_pressure'] - data['closing_pressure']
    data['efficiency_asymmetry'] = data['opening_efficiency'] - data['closing_efficiency']
    
    # Regime-Adaptive Signal Generation
    data['high_vol_alpha'] = 0.0
    data['normal_vol_alpha'] = 0.0
    data['low_vol_alpha'] = 0.0
    
    # High Volatility Regime
    volatility_signal = data['directional_bias'] * data['entropy_momentum'] * data['range_complexity'] * data['micro_meso_momentum'] * data['pressure_asymmetry']
    data['high_vol_alpha'] = volatility_signal * data['range_complexity'] * (data['opening_efficiency'] + data['closing_efficiency']) / 2
    
    # Normal Volatility Regime
    balance_signal = data['directional_bias'] * data['entropy_momentum'] * data['meso_range'] * data['efficiency_asymmetry']
    data['normal_vol_alpha'] = balance_signal * (data['micro_meso_momentum'] + data['meso_macro_momentum']) / 2 * data['volume'] / (data['volume'].rolling(window=5).mean() + 1e-10)
    
    # Low Volatility Regime
    breakout_signal = data['directional_bias'] * data['entropy_acceleration'] * data['macro_range'] * data['pressure_asymmetry']
    data['low_vol_alpha'] = breakout_signal * data['range_complexity'] * data['closing_efficiency']
    
    # Adaptive Alpha Integration
    def select_regime_alpha(row):
        if row['vol_regime'] == 'high':
            return row['high_vol_alpha']
        elif row['vol_regime'] == 'normal':
            return row['normal_vol_alpha']
        else:  # low volatility
            return row['low_vol_alpha']
    
    data['selected_alpha'] = data.apply(select_regime_alpha, axis=1)
    
    # Final Alpha with directional confirmation
    data['final_alpha'] = data['selected_alpha'] * np.sign(data['directional_bias']) * np.sign(data['entropy_momentum'])
    
    # Clean up and return
    alpha_series = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
