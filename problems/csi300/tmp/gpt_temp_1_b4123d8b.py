import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Fracture Asymmetry
    # Fracture Efficiency Asymmetry
    data['Immediate_Fracture_Efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['Medium_Fracture_Efficiency'] = np.abs(data['close'] - data['open'].shift(5)) / data['high'].rolling(5).apply(lambda x: (x - data.loc[x.index, 'low']).sum(), raw=False)
    data['Fracture_Efficiency_Divergence'] = data['Immediate_Fracture_Efficiency'] - data['Medium_Fracture_Efficiency']
    
    # Fracture Pressure Asymmetry
    data['Opening_Fracture_Pressure'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['Closing_Fracture_Pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['Upper_Fracture_Pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['Lower_Fracture_Pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    data['Combined_Fracture_Asymmetry'] = (data['Fracture_Efficiency_Divergence'] * 
                                         (data['Closing_Fracture_Pressure'] - data['Opening_Fracture_Pressure']) * 
                                         (data['Lower_Fracture_Pressure'] - data['Upper_Fracture_Pressure']))
    
    # Volume-Fracture Entanglement
    # Volume Asymmetry Dynamics
    data['Volume_Spike_Intensity'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3)
    data['Volume_Persistence'] = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(2))
    data['Flow_Intensity'] = (np.sign(data['close'] - data['close'].shift(1)) * 
                            (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1) * 
                            np.abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1))
    
    # Amount-Fracture Components
    data['Fracture_Trade_Size'] = data['amount'] / data['volume']
    data['Fracture_Trade_Momentum'] = (data['Fracture_Trade_Size'] / data['Fracture_Trade_Size'].shift(1))
    data['Fracture_Value_Pressure'] = data['Fracture_Trade_Size'] * (data['close'] - data['open']) / (data['high'] - data['low'])
    
    data['Volume_Fracture_Fusion'] = (data['Volume_Spike_Intensity'] * data['Volume_Persistence'] * 
                                    data['Flow_Intensity'] * data['Fracture_Value_Pressure'])
    
    # Asymmetric Momentum Structure
    # Multi-Scale Momentum Asymmetry
    data['Immediate_Momentum'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['Medium_term_Momentum'] = (data['close'] - data['close'].shift(5)) / data['high'].rolling(5).apply(lambda x: (x - data.loc[x.index, 'low']).sum(), raw=False)
    data['Momentum_Convergence'] = (np.sign(data['Immediate_Momentum']) * np.sign(data['Medium_term_Momentum']) / 
                                  (1 + np.abs(data['Immediate_Momentum'] - data['Medium_term_Momentum'])))
    
    # Volatility Asymmetry Detection
    data['Micro_Volatility'] = (data['high'] - data['low']) / data['close']
    data['Meso_Volatility'] = data['high'].rolling(5).apply(lambda x: (x - data.loc[x.index, 'low']).sum(), raw=False) / data['close'].rolling(5).sum()
    data['Volatility_Asymmetry'] = data['Micro_Volatility'] / data['Meso_Volatility']
    
    data['Asymmetric_Momentum'] = np.where(data['Volatility_Asymmetry'] > 1, 
                                         data['Medium_term_Momentum'], 
                                         data['Immediate_Momentum']) * data['Momentum_Convergence']
    
    # Breakout-Asymmetric Signals
    # Fracture Breakout Asymmetry
    data['Price_Breakout'] = ((data['close'] - data['low'].rolling(10).min()) / 
                            (data['high'].rolling(10).max() - data['low'].rolling(10).min()))
    data['Volume_Breakout'] = data['volume'] / data['volume'].rolling(10).max()
    data['Fracture_Breakout_Efficiency'] = (data['Price_Breakout'] * data['Volume_Breakout'] * 
                                          np.abs(data['close'] - data['open']) / (data['high'] - data['low']))
    
    # Gap-Fracture Asymmetry
    data['Fracture_Overnight_Gap'] = (data['open'] - data['close'].shift(1)) / np.abs(data['open'] - data['close'].shift(1))
    data['Fracture_Intraday_Recovery'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['Fracture_Gap_Asymmetry'] = -np.sign(data['Fracture_Overnight_Gap']) * data['Fracture_Intraday_Recovery']
    
    data['Breakout_Asymmetric_Confirmation'] = data['Fracture_Breakout_Efficiency'] * data['Fracture_Gap_Asymmetry']
    
    # Core Asymmetric Construction
    data['Fracture_Asymmetric_Foundation'] = data['Combined_Fracture_Asymmetry'] * data['Volume_Fracture_Fusion']
    data['Momentum_Enhanced_Asymmetry'] = data['Fracture_Asymmetric_Foundation'] * data['Asymmetric_Momentum']
    data['Breakout_Optimized_Asymmetry'] = data['Momentum_Enhanced_Asymmetry'] * data['Breakout_Asymmetric_Confirmation']
    
    # Regime-Adaptive Final Alpha
    # Volatility regime detection
    data['Volatility_Ratio'] = data['Micro_Volatility'] / data['Meso_Volatility']
    data['Expanding_Volatility'] = (data['high'].rolling(20).max() / data['low'].rolling(20).min() - 1)
    data['Contracting_Volatility'] = (np.abs(data['close'] - data['open']).rolling(5).sum() / 
                                    np.abs(data['close'] - data['open']).rolling(10).sum() - 1)
    
    # Regime selection
    high_vol_regime = data['Breakout_Optimized_Asymmetry'] * np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    low_vol_regime = data['Breakout_Optimized_Asymmetry'] * np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    expanding_regime = data['Breakout_Optimized_Asymmetry'] * data['Expanding_Volatility']
    contracting_regime = data['Breakout_Optimized_Asymmetry'] * data['Contracting_Volatility']
    
    # Final regime-adaptive signal
    regime_signal = np.where(data['Volatility_Ratio'] > 1.5, high_vol_regime,
                           np.where(data['Volatility_Ratio'] < 0.7, low_vol_regime,
                                  np.where(data['Expanding_Volatility'] > 0.02, expanding_regime,
                                         np.where(data['Contracting_Volatility'] < -0.01, contracting_regime,
                                                data['Breakout_Optimized_Asymmetry']))))
    
    # Final Alpha
    final_alpha = regime_signal * data['Fracture_Value_Pressure'] / (1 + np.abs(data['Fracture_Value_Pressure']))
    
    return final_alpha
