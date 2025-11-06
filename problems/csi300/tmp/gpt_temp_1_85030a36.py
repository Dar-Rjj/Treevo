import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Momentum Fracture & Asymmetry Framework
    # Intraday Momentum Patterns
    data['Morning_Momentum_Fracture'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    data['Afternoon_Momentum_Fracture'] = (data['close'] - data['low']) - (data['high'] - data['close'])
    data['Intraday_Fracture_Asymmetry'] = data['Morning_Momentum_Fracture'] - data['Afternoon_Momentum_Fracture']
    
    # Multi-Timeframe Momentum
    data['Short_term_Momentum_Fracture'] = (data['close'] - data['open']) / (data['close'].shift(3) - data['open'].shift(3)) - 1
    data['Medium_term_Momentum_Fracture'] = (data['close'] - data['open']) / (data['close'].shift(8) - data['open'].shift(8)) - 1
    data['Momentum_Fracture_Acceleration'] = data['Short_term_Momentum_Fracture'] - data['Medium_term_Momentum_Fracture']
    
    # Fracture-Momentum Integration
    data['Momentum_Asymmetry'] = (data['close'] - data['low']) / (data['high'] - data['low']) - 0.5
    data['Upper_Momentum_Intensity'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
    data['Lower_Momentum_Intensity'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])
    
    # Efficiency-Compression System
    # Price-Flow Efficiency
    data['Movement_Efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['Gap_Volatility_Efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['Absorption_Efficiency'] = (data['volume'] / abs(data['close'] - data['open'])) - (data['volume'].shift(1) / abs(data['close'].shift(1) - data['open'].shift(1)))
    
    # Volume-Compression Patterns
    data['Morning_Volume_Compression'] = data['volume'] * (data['high'] - data['open']) / data['amount']
    data['Afternoon_Volume_Compression'] = data['volume'] * (data['close'] - data['low']) / data['amount']
    data['Volume_Compression_Differential'] = data['Morning_Volume_Compression'] - data['Afternoon_Volume_Compression']
    
    # Efficiency-Compression Integration
    data['Strong_Efficiency_Conformation'] = ((data['Movement_Efficiency'] > 0.7) & (data['volume'] / data['volume'].shift(1) > 1)).astype(int)
    data['Efficiency_Enhanced_Compression'] = ((abs(data['Morning_Momentum_Fracture'] - data['Afternoon_Momentum_Fracture']) > 0.5) & (data['Strong_Efficiency_Conformation'] == 1)).astype(int)
    
    # Regime-Based Detection
    # Momentum Structure
    momentum_expansion = pd.Series(index=data.index, dtype=float)
    momentum_contraction = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 3:
            expansion_sum = 0
            contraction_sum = 0
            for j in range(4):
                current_momentum = data['close'].iloc[i-j] - data['open'].iloc[i-j]
                prev_momentum = data['close'].iloc[i-j-1] - data['open'].iloc[i-j-1]
                expansion_sum += max(0, current_momentum - prev_momentum)
                contraction_sum += max(0, prev_momentum - current_momentum)
            momentum_expansion.iloc[i] = expansion_sum
            momentum_contraction.iloc[i] = contraction_sum
    
    data['Momentum_Expansion'] = momentum_expansion
    data['Momentum_Contraction'] = momentum_contraction
    data['Momentum_Regime'] = (data['Momentum_Expansion'] - data['Momentum_Contraction']) / (data['Momentum_Expansion'] + data['Momentum_Contraction'] + 1e-8)
    
    # Fracture Regime
    data['Upper_Fracture_Regime'] = ((data['Intraday_Fracture_Asymmetry'] > 0) & (data['volume'] / data['volume'].shift(1) > 1)).astype(int)
    data['Lower_Fracture_Regime'] = ((data['Intraday_Fracture_Asymmetry'] < 0) & (data['volume'] / data['volume'].shift(1) < 1)).astype(int)
    data['Momentum_Regime_Detection'] = ((data['close'] - data['open']) > (data['close'].shift(5) - data['open'].shift(5))).astype(int)
    
    # Regime-Enhanced Signals
    data['High_Momentum_Regime'] = ((abs(data['Momentum_Regime']) > 0.4) | (data['Momentum_Regime_Detection'] == 1)).astype(int)
    data['Low_Momentum_Regime'] = ((abs(data['Momentum_Regime']) <= 0.4) & (data['Momentum_Regime_Detection'] == 0)).astype(int)
    
    # Multi-Scale Signal Construction
    # Short-term Components
    data['Short_term_Component_1'] = data['Momentum_Fracture_Acceleration'] * (data['volume'] / data['volume'].shift(1))
    data['Short_term_Component_2'] = data['Momentum_Asymmetry'] * data['Movement_Efficiency']
    data['Short_term_Component_3'] = data['Gap_Volatility_Efficiency'] * data['Volume_Compression_Differential']
    
    # Medium-term Components
    data['Medium_term_Component_1'] = abs(data['Morning_Momentum_Fracture'] - data['Afternoon_Momentum_Fracture']) * data['Momentum_Fracture_Acceleration']
    data['Medium_term_Component_2'] = data['Momentum_Regime_Detection'] * (data['volume'] / data['volume'].shift(1))
    data['Medium_term_Component_3'] = ((data['close'] - data['open']) / (data['close'].shift(2) - data['open'].shift(2)) - (data['close'] - data['open']) / (data['close'].shift(8) - data['open'].shift(8))) * data['Intraday_Fracture_Asymmetry']
    
    # Signal Validation
    data['Short_term_Signal'] = (data['Short_term_Component_1'] * data['Short_term_Component_2'] * 
                                data['Short_term_Component_3'])
    data['Medium_term_Signal'] = (data['Medium_term_Component_1'] * data['Medium_term_Component_2'])
    data['Signal_Alignment'] = np.sign(data['Short_term_Signal']) * np.sign(data['Medium_term_Signal'])
    
    # Adaptive Alpha Generation
    # High Momentum Alpha
    data['Volume_Confirmed_Fracture'] = ((abs(data['Morning_Momentum_Fracture'] - data['Afternoon_Momentum_Fracture']) > 0.5) & 
                                       (data['volume'] / data['volume'].shift(1) > 1.5)).astype(int)
    
    # Calculate rolling sum of Intraday Fracture Asymmetry
    data['Intraday_Fracture_Asymmetry_Sum'] = data['Intraday_Fracture_Asymmetry'].rolling(window=3, min_periods=1).sum()
    
    data['Upper_Fracture_Signal'] = ((data['Morning_Volume_Compression'] > data['Afternoon_Volume_Compression']) * 
                                   data['Upper_Fracture_Regime'] * data['Intraday_Fracture_Asymmetry_Sum'])
    data['Lower_Fracture_Signal'] = ((data['Morning_Volume_Compression'] < data['Afternoon_Volume_Compression']) * 
                                   data['Lower_Fracture_Regime'] * data['Intraday_Fracture_Asymmetry_Sum'])
    
    data['High_Momentum_Alpha'] = ((data['Volume_Confirmed_Fracture'] + data['Upper_Fracture_Signal'] + data['Lower_Fracture_Signal']) * 
                                 data['Short_term_Signal'] * data['Medium_term_Signal'] * data['Signal_Alignment'])
    
    # Low Momentum Alpha
    data['Efficiency_Fracture'] = ((abs(data['Morning_Momentum_Fracture'] - data['Afternoon_Momentum_Fracture']) > 0.5) & 
                                 (data['Strong_Efficiency_Conformation'] == 1)).astype(int)
    
    data['Mixed_Fracture_Signal'] = ((data['Morning_Volume_Compression'] > data['Afternoon_Volume_Compression']) * 
                                   ((data['Upper_Fracture_Regime'] == 0) & (data['Lower_Fracture_Regime'] == 0)) * 
                                   (data['volume'] / data['volume'].shift(1)))
    
    data['Low_Momentum_Alpha'] = ((data['Efficiency_Fracture'] + data['Mixed_Fracture_Signal']) * 
                                data['Volume_Compression_Differential'] * (data['volume'] / data['volume'].shift(1)) * 
                                data['Movement_Efficiency'] * data['Absorption_Efficiency'] * data['Signal_Alignment'])
    
    # Composite Alpha
    data['High_Momentum_Component'] = data['High_Momentum_Regime'] * data['High_Momentum_Alpha']
    data['Low_Momentum_Component'] = data['Low_Momentum_Regime'] * data['Low_Momentum_Alpha']
    data['Final_Alpha'] = data['High_Momentum_Component'] + data['Low_Momentum_Component']
    
    return data['Final_Alpha']
