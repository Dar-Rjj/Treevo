import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Microstructure Momentum Synthesis with Volatility Adaptation
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic technical indicators
    df['TR'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                   abs(df['low'] - df['close'].shift(1))))
    df['ATR_5'] = df['TR'].rolling(window=5).mean()
    
    # Price momentum calculations
    df['momentum_1d'] = df['close'] - df['close'].shift(1)
    df['momentum_3d'] = df['close'] - df['close'].shift(3)
    
    # Fractal Pressure Momentum components
    df['directional_pressure'] = ((df['high'] - df['close']) / 
                                 (df['close'] - df['low'] + 1e-8)) * (df['open'] - df['close']) * df['amount']
    
    # Fractal dimension approximation using price volatility
    df['fractal_dimension'] = df['TR'].rolling(window=5).std() / (df['ATR_5'] + 1e-8)
    
    # Fractal pressure persistence
    df['fractal_pressure_2d'] = df['directional_pressure'].rolling(window=2).mean()
    df['fractal_pressure_5d'] = df['directional_pressure'].rolling(window=5).mean()
    df['pressure_persistence_ratio'] = df['fractal_pressure_2d'] / (df['fractal_pressure_5d'] + 1e-8)
    
    # Fractal Range Efficiency Momentum
    df['range_efficiency'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)) * df['momentum_3d']
    
    # Fractal stability using range consistency
    df['range_std_3d'] = (df['high'] - df['low']).rolling(window=3).std()
    df['range_std_5d'] = (df['high'] - df['low']).rolling(window=5).std()
    df['fractal_stability'] = df['range_std_3d'] / (df['range_std_5d'] + 1e-8)
    
    df['efficiency_momentum'] = df['range_efficiency'] * df['fractal_stability']
    
    # Volume-Fractal Flow Dynamics
    df['value_density'] = (df['volume'] / (df['amount'] + 1e-8)) * df['close'] * df['momentum_1d']
    
    # Fractal coupling using volume-price relationship
    df['volume_price_corr_5d'] = df['volume'].rolling(window=5).corr(df['close'])
    df['fractal_coupling'] = abs(df['volume_price_corr_5d'])
    
    df['value_density_momentum'] = df['value_density'] * df['fractal_coupling']
    
    # Fractal Directional Flow
    df['microstructure_flow'] = (df['close'] - df['open']) * df['amount'] * df['directional_pressure']
    
    # Volatility-Adaptive Fractal Assessment
    df['volatility_regime'] = df['TR'] / (df['ATR_5'] + 1e-8)
    
    # Fractal efficiency in volatility context
    df['volatility_adjusted_efficiency'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)) * \
                                          (df['TR'] / (df['ATR_5'] + 1e-8))
    
    # Fractal persistence using autocorrelation
    df['price_autocorr_3d'] = df['close'].rolling(window=3).apply(lambda x: x.autocorr(), raw=False)
    df['fractal_persistence'] = abs(df['price_autocorr_3d'])
    
    df['adaptive_efficiency'] = df['volatility_adjusted_efficiency'] * df['fractal_persistence']
    
    # Fractal Divergence Momentum
    df['short_term_pressure'] = df['directional_pressure'].rolling(window=2).mean()
    df['long_term_pressure'] = df['directional_pressure'].rolling(window=5).mean()
    df['fractal_divergence'] = df['short_term_pressure'] - df['long_term_pressure']
    
    # Fractal divergence acceleration
    df['divergence_momentum'] = df['fractal_divergence'].diff(periods=1)
    
    # Multi-Dimensional Fractal Signal Integration
    
    # High volatility regime signals
    high_vol_mask = df['volatility_regime'] > 1.2
    df['high_vol_signal'] = df['directional_pressure'] * df['microstructure_flow'] * high_vol_mask
    
    # Low volatility regime signals  
    low_vol_mask = df['volatility_regime'] < 0.8
    df['low_vol_signal'] = df['adaptive_efficiency'] * df['value_density_momentum'] * low_vol_mask
    
    # Normal volatility regime signals
    normal_vol_mask = ~high_vol_mask & ~low_vol_mask
    df['normal_vol_signal'] = (df['pressure_persistence_ratio'] * df['efficiency_momentum'] * 
                              df['fractal_coupling']) * normal_vol_mask
    
    # Alpha Factor Synthesis
    df['fractal_pressure_momentum'] = (df['directional_pressure'] * df['pressure_persistence_ratio'] * 
                                      df['adaptive_efficiency'])
    
    df['fractal_value_momentum'] = (df['value_density_momentum'] * df['fractal_coupling'] * 
                                   df['fractal_persistence'])
    
    # Cross-Fractal Microstructure Convergence
    df['cross_fractal_convergence'] = (df['fractal_pressure_momentum'] * df['fractal_value_momentum'] * 
                                      (1 + df['divergence_momentum']))
    
    # Final alpha factor with regime adaptation
    alpha_factor = (df['high_vol_signal'].fillna(0) + 
                   df['low_vol_signal'].fillna(0) + 
                   df['normal_vol_signal'].fillna(0) + 
                   df['cross_fractal_convergence'].fillna(0))
    
    # Normalize the final factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20).mean()) / \
                   (alpha_factor.rolling(window=20).std() + 1e-8)
    
    return alpha_factor
