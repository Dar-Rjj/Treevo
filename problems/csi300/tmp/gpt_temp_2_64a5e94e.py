import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Acceleration Asymmetry
    data['Upper_Price_Acceleration'] = (data['high'] - data['high'].shift(1)) / (data['high'].shift(1) - data['high'].shift(2))
    data['Lower_Price_Acceleration'] = (data['low'] - data['low'].shift(1)) / (data['low'].shift(1) - data['low'].shift(2))
    data['Asymmetric_Fracture'] = ((data['Upper_Price_Acceleration'].abs() > 2.0) | (data['Lower_Price_Acceleration'].abs() > 2.0)).astype(int)
    
    # Multi-Timeframe Range Fracture
    data['Fast_Range_Asymmetry'] = (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3)) - 1
    data['Medium_Range_Asymmetry'] = (data['high'] - data['low']) / (data['high'].shift(8) - data['low'].shift(8)) - 1
    data['Range_Fracture_Divergence'] = data['Fast_Range_Asymmetry'] - data['Medium_Range_Asymmetry']
    
    # Volatility-Asymmetric Integration
    data['Upper_Volatility_Efficiency'] = (data['high'] - data['open']) / (data['high'] - data['low'])
    data['Lower_Volatility_Efficiency'] = (data['open'] - data['low']) / (data['high'] - data['low'])
    data['Volatility_Asymmetry'] = data['Upper_Volatility_Efficiency'] - data['Lower_Volatility_Efficiency']
    
    # Asymmetric Volume Flow
    data['Upper_Volume_Flow'] = data['amount'] * (data['high'] - data['close']) / (data['high'] - data['low'])
    data['Lower_Volume_Flow'] = data['amount'] * (data['close'] - data['low']) / (data['high'] - data['low'])
    data['Volume_Flow_Asymmetry'] = data['Upper_Volume_Flow'] - data['Lower_Volume_Flow']
    
    # Volume-Price Convergence Dynamics
    data['Price_Volatility_Convergence'] = ((data['close'] - data['open']) / (data['high'] - data['low']) - 
                                          (data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)))
    data['Volume_Volatility_Convergence'] = (data['volume'] / (data['high'] - data['low']) - 
                                           data['volume'].shift(1) / (data['high'].shift(1) - data['low'].shift(1)))
    data['Convergence_Alignment'] = np.sign(data['Price_Volatility_Convergence']) * np.sign(data['Volume_Volatility_Convergence'])
    
    # Volume-Convergence Fracture
    data['Volume_Convergence_Fracture'] = (data['Asymmetric_Fracture'] & 
                                         (np.sign(data['Volume_Flow_Asymmetry']) * np.sign(data['Range_Fracture_Divergence']) < 0)).astype(int)
    
    # Fractal Momentum Components
    data['Short_term_Price_Momentum'] = ((data['close'] / data['close'].shift(3) - 1) * 
                                       ((data['close'] - data['open']) / (data['high'] - data['low'])))
    data['Medium_term_Price_Momentum'] = ((data['close'] / data['close'].shift(8) - 1) * 
                                        ((data['close'] - data['open']) / (data['high'] - data['low'])))
    data['Fractal_Momentum_Divergence'] = data['Short_term_Price_Momentum'] - data['Medium_term_Price_Momentum']
    
    # Directional Momentum Framework
    bullish_pressure = pd.Series(index=data.index, dtype=float)
    bearish_pressure = pd.Series(index=data.index, dtype=float)
    for i in range(5):
        bullish_pressure += np.maximum(0, data['close'].shift(i) - data['close'].shift(i+1))
        bearish_pressure += np.maximum(0, data['close'].shift(i+1) - data['close'].shift(i))
    data['Directional_imbalance'] = (bullish_pressure - bearish_pressure) / (bullish_pressure + bearish_pressure + 1e-8)
    
    # Momentum-Convergence Integration
    data['Fracture_momentum_alignment'] = (np.sign(data['Upper_Price_Acceleration'] - data['Lower_Price_Acceleration']) * 
                                         np.sign(data['Directional_imbalance']))
    data['Volume_momentum_convergence'] = np.sign(data['Volume_Flow_Asymmetry']) * np.sign(data['Fractal_Momentum_Divergence'])
    data['Multi_scale_momentum_convergence'] = np.sign(data['Short_term_Price_Momentum']) * np.sign(data['Medium_term_Price_Momentum'])
    
    # Asymmetric Volatility Components
    data['Upper_True_Range'] = np.maximum(data['high'] - data['close'].shift(1), data['high'] - data['low'])
    data['Lower_True_Range'] = np.maximum(data['close'].shift(1) - data['low'], data['high'] - data['low'])
    data['Asymmetric_Volatility_Ratio'] = data['Upper_True_Range'] / (data['Lower_True_Range'] + 1e-8)
    
    # Efficiency Dynamics
    data['Price_movement_efficiency'] = (data['close'] - data['close'].shift(1)).abs() / (data['high'] - data['low'])
    data['Volume_surge'] = data['volume'] / (data['volume'].rolling(window=5).mean())
    data['Volume_weighted_efficiency'] = data['Price_movement_efficiency'] * data['Volume_surge']
    
    # Volatility-Convergence Integration
    data['Volatility_scaled_fracture'] = data['Asymmetric_Fracture'] * data['Asymmetric_Volatility_Ratio']
    data['Efficiency_convergence_alignment'] = np.sign(data['Price_movement_efficiency']) * np.sign(data['Convergence_Alignment'])
    data['Volatility_enhanced_convergence'] = data['Volatility_Asymmetry'] * data['Volume_weighted_efficiency']
    
    # Fracture-Convergence Alignment
    data['Fracture_direction'] = np.sign(data['Upper_Price_Acceleration'] - data['Lower_Price_Acceleration'])
    data['Fracture_convergence_alignment'] = data['Fracture_direction'] * np.sign(data['Convergence_Alignment'])
    
    # Volume-Efficiency Convergence
    data['Volume_acceleration'] = data['volume'] / data['volume'].shift(1) - 1
    data['Efficiency_acceleration'] = data['Price_movement_efficiency'] / data['Price_movement_efficiency'].shift(1) - 1
    data['Volume_efficiency_convergence'] = np.sign(data['Volume_acceleration']) * np.sign(data['Efficiency_acceleration'])
    
    # Divergence Strength Assessment
    data['Fracture_strength'] = data['Range_Fracture_Divergence'].abs() * data['Volume_Flow_Asymmetry']
    data['Momentum_convergence_strength'] = data['Directional_imbalance'].abs() * data['Convergence_Alignment'].abs()
    data['Combined_convergence_strength'] = data['Fracture_strength'] + data['Momentum_convergence_strength']
    
    # Multi-Timeframe Signal Integration
    data['Short_term_signal'] = data['Asymmetric_Fracture'] * data['Volume_Flow_Asymmetry'] * data['Convergence_Alignment']
    data['Medium_term_signal'] = data['Range_Fracture_Divergence'] * data['Volatility_Asymmetry'] * data['Fractal_Momentum_Divergence']
    data['Signal_convergence'] = np.sign(data['Short_term_signal']) * np.sign(data['Medium_term_signal'])
    
    # Adaptive Convergence Alpha Construction
    high_fracture_mask = (data['Asymmetric_Fracture'] == 1) & (data['Volume_Convergence_Fracture'] == 1)
    
    # High Fracture-Convergence Regime
    fracture_convergence_factor = data['Fracture_convergence_alignment'] * data['Volume_momentum_convergence']
    volatility_efficiency_factor = data['Volume_weighted_efficiency'] * data['Volatility_enhanced_convergence']
    high_fracture_alpha = fracture_convergence_factor * volatility_efficiency_factor * data['Volatility_scaled_fracture']
    
    # Low Fracture-Convergence Regime
    range_convergence_factor = data['Range_Fracture_Divergence'] * data['Price_movement_efficiency']
    volume_persistence_factor = data['Volume_surge'] * data['Volume_Flow_Asymmetry']
    low_fracture_alpha = range_convergence_factor * volume_persistence_factor * data['Convergence_Alignment']
    
    # Final alpha signal
    data['Final_alpha_signal'] = np.where(high_fracture_mask, high_fracture_alpha, low_fracture_alpha)
    
    # Pattern Persistence Metrics
    data['Fracture_convergence_persistence'] = data['Fracture_convergence_alignment'].rolling(window=3).apply(lambda x: (x > 0).sum())
    data['Volume_efficiency_persistence'] = data['Volume_efficiency_convergence'].rolling(window=3).apply(lambda x: (x > 0).sum())
    data['Signal_consistency'] = data['Fracture_convergence_persistence'] * data['Volume_efficiency_persistence']
    
    # Regime Stability Framework
    data['Fracture_regime_stability'] = data['Asymmetric_Fracture'].rolling(window=5).apply(lambda x: (x == x.iloc[0]).sum())
    data['Convergence_regime_stability'] = data['Convergence_Alignment'].rolling(window=5).apply(lambda x: (x > 0).sum())
    data['Regime_transition_risk'] = ((data['Fracture_regime_stability'] - 3).abs() > 2) | ((data['Convergence_regime_stability'] - 3).abs() > 2)
    
    # Quality-Adjusted Convergence Alpha
    data['Signal_strength'] = data['Final_alpha_signal'].abs() * data['Signal_consistency']
    data['Risk_adjustment'] = 1 / (1 + data['Regime_transition_risk'].astype(int))
    data['Quality_adjusted_alpha'] = (data['Final_alpha_signal'] * data['Risk_adjustment'] * 
                                    (1 - 0.1 * (data['Fracture_convergence_alignment'] - 1).abs()))
    
    return data['Quality_adjusted_alpha']
