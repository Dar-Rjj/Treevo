import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fracture-Pressure Regime Identification
    # Fracture Components
    data['Intraday_Fracture'] = (data['high'] - data['low']) / data['close']
    data['Overnight_Fracture'] = abs(data['open'] / data['close'].shift(1) - 1)
    data['Total_Fracture'] = data['Intraday_Fracture'] + data['Overnight_Fracture']
    
    # Pressure Components
    data['Opening_Pressure'] = (data['open'] - (data['high'].shift(1) + data['low'].shift(1))/2) / ((data['high'].shift(1) - data['low'].shift(1))/2 + 1e-8)
    data['Closing_Pressure'] = (data['close'] - (data['high'] + data['low'])/2) / ((data['high'] - data['low'])/2 + 1e-8)
    data['Pressure_Persistence'] = np.sign(data['Opening_Pressure']) * np.sign(data['Closing_Pressure'])
    
    # Regime Classification
    fracture_ma = data['Total_Fracture'].rolling(window=20, min_periods=1).mean()
    data['High_Fracture_Regime'] = data['Total_Fracture'] > fracture_ma
    data['Low_Fracture_Regime'] = data['Total_Fracture'] < fracture_ma
    data['Normal_Fracture_Regime'] = ~(data['High_Fracture_Regime'] | data['Low_Fracture_Regime'])
    
    # Asymmetric Flow Dynamics
    # Volume-Flow Asymmetry
    data['Volume_Momentum'] = (data['volume'] / data['volume'].shift(1) - 1) * (data['volume'].shift(1) / data['volume'].shift(2) - 1)
    data['Amount_Volume_Coherence'] = (data['amount'] / data['amount'].shift(1) - 1) * (data['volume'] / data['volume'].shift(1) - 1)
    
    # Price-Fracture Asymmetry
    data['Upside_Volatility'] = (data['high'] - data['open']) / (data['open'] + 1e-8)
    data['Downside_Volatility'] = (data['open'] - data['low']) / (data['open'] + 1e-8)
    data['Volatility_Asymmetry'] = data['Upside_Volatility'] / (data['Downside_Volatility'] + 1e-8)
    
    # Gap-Flow Integration
    # Gap Analysis
    data['Gap_Size'] = data['open'] / data['close'].shift(1) - 1
    
    # Gap Persistence (count sign consistency from t-4 to t)
    gap_sign = np.sign(data['Gap_Size'])
    data['Gap_Persistence'] = 0
    for i in range(4, len(data)):
        if i >= 4:
            window_signs = gap_sign.iloc[i-4:i+1]
            if len(window_signs) == 5:
                data.loc[data.index[i], 'Gap_Persistence'] = (window_signs == window_signs.iloc[-1]).sum()
    
    # Fracture-Flow Interaction
    data['Gap_Absorption'] = (data['close'] - data['open']) / (data['close'].shift(1) - data['open'] + 1e-8)
    data['Fracture_Response'] = data['Total_Fracture'] * data['Gap_Absorption']
    
    # Cross-Timeframe Coherence
    # Return Components
    data['Short_term_Return'] = data['close'] / data['close'].shift(3) - 1
    data['Medium_term_Return'] = data['close'] / data['close'].shift(10) - 1
    data['Coherence_Ratio'] = data['Short_term_Return'] / (data['Medium_term_Return'] + 1e-8)
    data['Fractal_Coherence'] = data['Coherence_Ratio'] * data['Total_Fracture']
    
    # Momentum Filtering
    # Multi-Scale Momentum
    data['Short_term_Momentum'] = data['close'] / data['close'].shift(5) - 1
    data['Medium_term_Momentum'] = data['close'] / data['close'].shift(20) - 1
    data['Momentum_Divergence'] = data['Short_term_Momentum'] - data['Medium_term_Momentum']
    
    # Fracture Efficiency
    true_range_1 = data['high'] - data['low']
    true_range_2 = abs(data['high'] - data['close'].shift(1))
    true_range_3 = abs(data['low'] - data['close'].shift(1))
    data['True_Range'] = np.maximum(true_range_1, np.maximum(true_range_2, true_range_3))
    data['Fracture_Efficiency'] = abs(data['close'] - data['close'].shift(1)) / (data['True_Range'] + 1e-8)
    
    # Alpha Synthesis
    # Core Signal Construction
    data['Base_Signal'] = data['Momentum_Divergence'] * data['Fractal_Coherence']
    data['Pressure_Signal'] = data['Opening_Pressure'] * data['Closing_Pressure']
    data['Enhanced_Signal'] = data['Base_Signal'] * data['Pressure_Signal'] * data['Volatility_Asymmetry']
    
    # Dynamic Component Weighting
    data['Fracture_Strength'] = data['Total_Fracture'] * data['Gap_Persistence']
    data['Pressure_Strength'] = abs(data['Opening_Pressure']) + abs(data['Closing_Pressure'])
    data['Flow_Strength'] = abs(data['Volume_Momentum']) + abs(data['Amount_Volume_Coherence'])
    data['Adaptive_Weight'] = data['Fracture_Strength'] / (data['Fracture_Strength'] + data['Pressure_Strength'] + data['Flow_Strength'] + 1e-8)
    
    # Final Alpha Output
    alpha = pd.Series(index=data.index, dtype=float)
    
    # High Fracture Alpha
    high_fracture_mask = data['High_Fracture_Regime']
    alpha[high_fracture_mask] = (data['Enhanced_Signal'] * data['Adaptive_Weight'] * data['Fracture_Response'])[high_fracture_mask]
    
    # Low Fracture Alpha
    low_fracture_mask = data['Low_Fracture_Regime']
    alpha[low_fracture_mask] = (data['Enhanced_Signal'] * data['Adaptive_Weight'] * data['Pressure_Persistence'])[low_fracture_mask]
    
    # Normal Fracture Alpha
    normal_fracture_mask = data['Normal_Fracture_Regime']
    alpha[normal_fracture_mask] = (data['Enhanced_Signal'] * data['Adaptive_Weight'] * data['Fracture_Efficiency'])[normal_fracture_mask]
    
    return alpha
