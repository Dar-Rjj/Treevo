import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate True Range
    data['TR'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    
    # Volatility Asymmetry based on price movement direction
    data['Volatility_Asymmetry'] = np.where(data['close'] > data['open'], 
                                           (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-8),
                                           (data['close'] - data['low']) / (data['high'] - data['close'] + 1e-8))
    data['Volatility_Asymmetry'] = np.clip(data['Volatility_Asymmetry'], 0.1, 10)
    
    # Microstructure Persistence based on volume pattern
    data['Microstructure_Persistence'] = data['amount'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 5 and not np.isnan(x).any() else 0
    )
    
    # Asymmetric Gap Components
    data['Asymmetric_Gap_Ratio'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * data['Volatility_Asymmetry']
    data['Asymmetric_Gap_Adjusted_Volatility'] = (abs(data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * data['Volatility_Asymmetry']
    data['Asymmetric_Volatility_Normalized_Gap'] = ((data['close'] - data['open']) / (data['TR'] + 1e-8)) * data['Volatility_Asymmetry']
    data['Asymmetric_Gap_Fractal_Momentum'] = ((data['open'] - (data['high'].shift(1) + data['low'].shift(1))/2) / 
                                              ((data['high'].shift(1) - data['low'].shift(1))/2 + 1e-8)) * \
                                             (data['amount'] / data['amount'].shift(1)) * data['Microstructure_Persistence']
    
    # Asymmetric Volume-Volatility Momentum
    data['Asymmetric_Volume_Acceleration'] = (data['amount'] / data['amount'].shift(1) - 1) * (1 - data['Asymmetric_Gap_Ratio'])
    data['Asymmetric_Volume_Volatility_Momentum'] = ((data['amount'] / data['amount'].shift(1)) * 
                                                    ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * 
                                                    data['Volatility_Asymmetry'])
    data['Asymmetric_Volatility_Adjusted_Momentum'] = ((data['close'] - data['close'].shift(1)) / 
                                                      (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * data['Volatility_Asymmetry']
    data['Asymmetric_Volatility_Efficiency_Momentum'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * \
                                                       ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * \
                                                       data['Microstructure_Persistence']
    
    # Asymmetric Coherence Signals
    data['Asymmetric_Gap_Flow_Alignment'] = np.sign(data['Asymmetric_Gap_Fractal_Momentum']) * \
                                           np.sign(data['Asymmetric_Volume_Acceleration']) * \
                                           data['Microstructure_Persistence']
    
    # Rolling correlations for Volume-Fractal Coherence
    data['Volume_Fractal_Corr_10'] = data['amount'].rolling(window=10).corr(data['TR'])
    data['Volume_Fractal_Corr_10_5'] = data['Volume_Fractal_Corr_10'].shift(5)
    data['Asymmetric_Volume_Fractal_Coherence'] = (data['Volume_Fractal_Corr_10'] - data['Volume_Fractal_Corr_10_5']) * data['Volatility_Asymmetry']
    
    data['Asymmetric_Volatility_Momentum_Coherence'] = data['Asymmetric_Gap_Ratio'] * \
                                                      data['Asymmetric_Volume_Fractal_Coherence'] * \
                                                      data['Asymmetric_Gap_Flow_Alignment']
    
    # Asymmetric Price-Level Microstructure
    data['Asymmetric_Opening_Rejection'] = ((data['high'] - np.maximum(data['open'], data['close'])) / 
                                           (data['high'] - data['low'] + 1e-8)) * data['Volatility_Asymmetry']
    data['Asymmetric_Closing_Pressure'] = ((np.minimum(data['open'], data['close']) - data['low']) / 
                                          (data['high'] - data['low'] + 1e-8)) * data['Volatility_Asymmetry']
    data['Asymmetric_Support_Volatility_Intensity'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * \
                                                     ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * \
                                                     data['Volatility_Asymmetry']
    data['Asymmetric_Resistance_Volatility_Pressure'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) * \
                                                       ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * \
                                                       data['Volatility_Asymmetry']
    
    # Asymmetric Amount-Microstructure Dynamics
    data['Asymmetric_Amount_Flow_Intensity'] = (data['amount'] / (data['high'] - data['low'] + 1e-8)) * data['Volatility_Asymmetry']
    
    # Rolling correlation for Volume-Amount Divergence
    data['Volume_Amount_Corr'] = data['amount'].rolling(window=5).corr(data['volume'])
    data['Asymmetric_Volume_Amount_Divergence'] = data['Volume_Amount_Corr'] * data['Asymmetric_Volume_Acceleration'] * \
                                                 (data['amount'] / data['amount'].shift(1) - 1) * data['Microstructure_Persistence']
    
    data['Asymmetric_Microstructure_Pressure'] = (data['Asymmetric_Closing_Pressure'] - data['Asymmetric_Opening_Rejection']) * \
                                                data['Asymmetric_Volume_Acceleration']
    
    # Asymmetric Volatility Regime Signals
    data['MA_TR_10'] = data['TR'].rolling(window=10).mean()
    data['Asymmetric_Volatility_Breakout'] = (data['TR'] / data['MA_TR_10'] - 1) * data['Volatility_Asymmetry']
    data['Asymmetric_Volatility_Compression'] = (1 - data['TR'] / data['MA_TR_10']) * (1 - data['Volatility_Asymmetry'])
    data['Asymmetric_True_Range_Momentum'] = ((data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1))) * \
                                            data['Volatility_Asymmetry']
    
    # Composite Signal Construction
    data['Asymmetric_Volatility_Weighted_Gap'] = data['Asymmetric_Volatility_Normalized_Gap'] * data['Asymmetric_Volume_Acceleration']
    data['Asymmetric_Gap_Breakout_Intensity'] = data['Asymmetric_Gap_Ratio'] * data['Asymmetric_Volatility_Breakout']
    data['Asymmetric_Volatility_Efficiency_Composite'] = data['Asymmetric_Volatility_Efficiency_Momentum'] * data['Asymmetric_Gap_Ratio']
    data['Asymmetric_Support_Resistance_Momentum'] = (data['Asymmetric_Support_Volatility_Intensity'] - 
                                                     data['Asymmetric_Resistance_Volatility_Pressure']) * \
                                                     data['Asymmetric_Volume_Acceleration']
    
    data['Asymmetric_Rejection_Adjusted_Gap'] = ((data['close'] - (data['high'] + data['low'])/2) / 
                                                (data['high'] - data['low'] + 1e-8)) * \
                                               (1 - abs(data['Asymmetric_Opening_Rejection'] - data['Asymmetric_Closing_Pressure']))
    data['Asymmetric_Volume_Coherence_Momentum'] = data['Asymmetric_Volume_Volatility_Momentum'] * data['Asymmetric_Volume_Fractal_Coherence']
    data['Asymmetric_Microstructure_Pressure_Momentum'] = data['Asymmetric_Microstructure_Pressure'] * data['Asymmetric_Gap_Ratio']
    
    # Regime-Adaptive Components
    data['High_Asymmetric_Volatility_Component'] = data['Asymmetric_Volatility_Weighted_Gap'] * \
                                                  data['Asymmetric_Volatility_Breakout'] * \
                                                  data['Asymmetric_Gap_Flow_Alignment']
    data['Low_Asymmetric_Volatility_Component'] = data['Asymmetric_Volatility_Efficiency_Composite'] * \
                                                 data['Asymmetric_Volatility_Compression'] * \
                                                 (1 - abs(data['Asymmetric_Opening_Rejection'] - data['Asymmetric_Closing_Pressure']))
    data['Asymmetric_Transition_Component'] = data['Asymmetric_Support_Resistance_Momentum'] * \
                                             abs(data['Asymmetric_True_Range_Momentum']) * \
                                             data['Asymmetric_Volume_Fractal_Coherence']
    data['Asymmetric_Coherence_Component'] = data['Asymmetric_Volume_Coherence_Momentum'] * \
                                            data['Asymmetric_Gap_Flow_Alignment'] * \
                                            data['Asymmetric_Gap_Ratio']
    
    # Dynamic Asymmetric Integration
    data['Asymmetric_Volatility_Regime_Strength'] = abs(data['Asymmetric_Volatility_Breakout']) + abs(data['Asymmetric_Volatility_Compression'])
    data['Asymmetric_Gap_Efficiency_Strength'] = data['Asymmetric_Gap_Ratio'] * abs(data['Asymmetric_Gap_Fractal_Momentum'])
    data['Asymmetric_Volume_Coherence_Strength'] = data['Asymmetric_Volume_Fractal_Coherence'] * data['Asymmetric_Gap_Flow_Alignment']
    data['Asymmetric_Microstructure_Strength'] = 1 - abs(data['Asymmetric_Opening_Rejection'] - data['Asymmetric_Closing_Pressure'])
    
    # Regime detection for weighting
    high_vol = data['Asymmetric_Volatility_Breakout'] > 0.1
    high_coh = (data['Asymmetric_Volume_Fractal_Coherence'] > 0.1) & (data['Asymmetric_Gap_Flow_Alignment'] > 0)
    efficient_gap = data['Asymmetric_Gap_Ratio'] > 0.6
    
    data['Asymmetric_Regime_Weight'] = 0.8  # Default weight
    
    # Apply regime-specific weights
    data.loc[high_vol & high_coh, 'Asymmetric_Regime_Weight'] = 1.2 * data['Asymmetric_Volume_Coherence_Strength']
    data.loc[high_vol & efficient_gap, 'Asymmetric_Regime_Weight'] = 1.1 * data['Asymmetric_Gap_Efficiency_Strength']
    data.loc[~high_vol & high_coh, 'Asymmetric_Regime_Weight'] = 1.3 * data['Asymmetric_Volume_Coherence_Strength']
    data.loc[efficient_gap & (data['Asymmetric_True_Range_Momentum'].abs() > 0.05), 'Asymmetric_Regime_Weight'] = 1.0 * data['Asymmetric_Gap_Efficiency_Strength']
    
    # Weighted components
    data['Asymmetric_Volatility_Component'] = data['High_Asymmetric_Volatility_Component'] * data['Asymmetric_Volatility_Regime_Strength']
    data['Asymmetric_Efficiency_Component'] = data['Low_Asymmetric_Volatility_Component'] * data['Asymmetric_Gap_Efficiency_Strength']
    data['Asymmetric_Transition_Component_Weighted'] = data['Asymmetric_Transition_Component'] * data['Asymmetric_Microstructure_Strength']
    data['Asymmetric_Coherence_Component_Weighted'] = data['Asymmetric_Coherence_Component'] * data['Asymmetric_Volume_Coherence_Strength']
    
    # Base Alpha
    data['Base_Asymmetric_Alpha'] = (data['Asymmetric_Volatility_Component'] + 
                                    data['Asymmetric_Efficiency_Component'] + 
                                    data['Asymmetric_Transition_Component_Weighted'] + 
                                    data['Asymmetric_Coherence_Component_Weighted'])
    
    # Final Enhancement Factors
    data['Asymmetric_Fractal_Momentum_Alignment'] = np.sign(data['Asymmetric_Gap_Fractal_Momentum']) * \
                                                   np.sign(data['Asymmetric_Volume_Acceleration']) * \
                                                   data['Microstructure_Persistence']
    data['Asymmetric_Volatility_Coherence_Consistency'] = data['Asymmetric_Volume_Fractal_Coherence'] * data['Asymmetric_Volatility_Efficiency_Momentum']
    data['Asymmetric_Microstructure_Consistency'] = (1 - abs(data['Asymmetric_Opening_Rejection'] - data['Asymmetric_Closing_Pressure'])) * data['Asymmetric_Gap_Ratio']
    
    data['Efficient_Asymmetric_Regime_Boost'] = 1 + (data['Asymmetric_Gap_Ratio'] * 0.5)
    data['High_Asymmetric_Coherence_Multiplier'] = 1 + (data['Asymmetric_Volume_Fractal_Coherence'] * 0.3)
    data['Asymmetric_Volatility_Regime_Enhancement'] = 1 + (abs(data['Asymmetric_Volatility_Breakout']) * 0.4)
    data['Asymmetric_Microstructure_Enhancement'] = 1 + (data['Asymmetric_Microstructure_Consistency'] * 0.2)
    
    # Final Alpha
    data['Final_Asymmetric_Alpha'] = (data['Base_Asymmetric_Alpha'] * 
                                     data['Asymmetric_Regime_Weight'] * 
                                     data['Asymmetric_Fractal_Momentum_Alignment'] * 
                                     data['Efficient_Asymmetric_Regime_Boost'] * 
                                     data['High_Asymmetric_Coherence_Multiplier'] * 
                                     data['Asymmetric_Volatility_Regime_Enhancement'] * 
                                     data['Asymmetric_Microstructure_Enhancement'])
    
    return data['Final_Asymmetric_Alpha']
