import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Fractal Volatility Regime Classification
    data['Current_Volatility'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    
    # Fractal Volatility Pattern
    data['High_max_prev2'] = data['high'].shift(1).combine(data['high'].shift(2), max)
    data['Low_min_prev2'] = data['low'].shift(1).combine(data['low'].shift(2), min)
    data['Fractal_Volatility_Pattern'] = (data['high'] - data['High_max_prev2']) - (data['low'] - data['Low_min_prev2'])
    
    # Volume-Confirmed Volatility
    data['Volume_Confirmed_Volatility'] = data['Current_Volatility'] * (data['volume'] / data['volume'].shift(1) - 1)
    
    # Regime Classification
    vol_threshold = data['Volume_Confirmed_Volatility'].rolling(window=20, min_periods=1).median()
    data['High_Vol_Regime'] = (data['Volume_Confirmed_Volatility'] > vol_threshold).astype(int)
    
    # Fractal Efficiency Dynamics
    data['Fractal_Range_Efficiency'] = (data['close'] - data['open']) / (
        data['high'] - data['High_max_prev2'] + data['Low_min_prev2'] - data['low']
    ).replace(0, np.nan)
    
    data['Volume_Weighted_Gap_Efficiency'] = (
        (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    ) * (data['volume'] / data['volume'].shift(1) - 1)
    
    data['Fractal_Directional_Efficiency'] = np.sign(data['close'] - data['open']) * data['Fractal_Range_Efficiency']
    data['Efficiency_Volume_Divergence'] = data['Fractal_Range_Efficiency'] - data['Volume_Weighted_Gap_Efficiency']
    
    # Multi-Scale Fractal Momentum
    data['Fractal_Short_term_Pressure'] = (
        (data['close'] - data['open']) / data['close'].shift(1)
    ) * (data['high'] - data['High_max_prev2'])
    
    data['Volume_Confirmed_Medium_Momentum'] = (
        (data['close'] / data['close'].shift(10) - 1)
    ) * (data['volume'] / data['volume'].shift(1) - 1)
    
    data['Fractal_Long_term_Trend'] = (
        (data['close'] / data['close'].shift(60) - 1)
    ) * data['Fractal_Volatility_Pattern']
    
    data['Fractal_Momentum_Acceleration'] = (
        data['Fractal_Short_term_Pressure'] - data['Volume_Confirmed_Medium_Momentum']
    )
    
    data['Volume_Fractal_Momentum_Cascade'] = (
        data['Fractal_Short_term_Pressure'] * 
        data['Volume_Confirmed_Medium_Momentum'] * 
        data['Fractal_Long_term_Trend']
    )
    
    # Volume-Fractal Confirmation
    data['Fractal_Volume_Efficiency'] = (
        data['amount'] / (data['volume'] * data['close'])
    ) * (data['volume'] / data['volume'].shift(1) - 1)
    
    data['Volume_Fractal_Concentration'] = (
        data['volume'] / data['volume'].shift(20)
    ) * data['Fractal_Volatility_Pattern']
    
    data['Fractal_Volume_Acceleration'] = (
        (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    ) * (data['volume'] / data['volume'].shift(1) - 1)
    
    data['Fractal_Price_Volume_Alignment'] = (
        np.sign(data['Fractal_Momentum_Acceleration']) * 
        np.sign(data['Fractal_Volume_Acceleration'])
    )
    
    data['Fractal_Efficiency_Volume_Synergy'] = (
        data['Fractal_Range_Efficiency'] * data['Fractal_Volume_Efficiency']
    )
    
    # Gap-Fractal Momentum Integration
    data['Fractal_Overnight_Gap'] = (
        (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    ) * (data['high'] - data['High_max_prev2'])
    
    data['Fractal_Gap_Sustainability'] = (
        (data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1))
    ) * data['Fractal_Volatility_Pattern']
    
    data['Fractal_Gap_Volume_Confirmation'] = (
        data['volume'] / data['volume'].shift(5)
    ) * (data['volume'] / data['volume'].shift(1) - 1)
    
    data['Fractal_Gap_Fill_Momentum'] = (
        data['Fractal_Gap_Sustainability'] * data['Fractal_Momentum_Acceleration']
    )
    
    data['Volume_Fractal_Weighted_Gap'] = (
        data['Fractal_Overnight_Gap'] * data['Fractal_Gap_Volume_Confirmation']
    )
    
    # Fractal Regime-Adaptive Factor Construction
    # High Volatility Fractal Regime Factors
    data['Fractal_Efficiency_Momentum_Core'] = (
        data['Fractal_Range_Efficiency'] * data['Fractal_Momentum_Acceleration']
    )
    
    data['Volume_Fractal_Confirmed_Gap'] = (
        data['Fractal_Gap_Fill_Momentum'] * data['Fractal_Volume_Efficiency']
    )
    
    data['Fractal_Range_Constrained_Momentum'] = (
        data['Fractal_Momentum_Acceleration'] * data['Fractal_Directional_Efficiency']
    )
    
    data['High_Volatility_Fractal_Composite'] = (
        data['Fractal_Efficiency_Momentum_Core'] * 
        data['Volume_Fractal_Confirmed_Gap'] * 
        data['Fractal_Range_Constrained_Momentum']
    )
    
    # Low Volatility Fractal Regime Factors
    data['Volume_Fractal_Weighted_Trend'] = (
        data['Volume_Fractal_Weighted_Gap'] * data['Fractal_Long_term_Trend']
    )
    
    data['Fractal_Efficiency_Persistence'] = (
        (data['Fractal_Range_Efficiency'] / data['Fractal_Range_Efficiency'].shift(5) - 1) *
        data['Fractal_Price_Volume_Alignment']
    )
    
    data['Fractal_Range_Adapted_Momentum'] = (
        data['Fractal_Momentum_Acceleration'] * data['Fractal_Volatility_Pattern']
    )
    
    data['Low_Volatility_Fractal_Composite'] = (
        data['Volume_Fractal_Weighted_Trend'] * 
        data['Fractal_Efficiency_Persistence'] * 
        data['Fractal_Range_Adapted_Momentum']
    )
    
    # Fractal Regime Transition Handling
    data['Fractal_Volatility_Proximity'] = abs(data['Volume_Confirmed_Volatility'] - vol_threshold)
    max_proximity = data['Fractal_Volatility_Proximity'].rolling(window=20, min_periods=1).max()
    data['Fractal_Smooth_Transition'] = data['Fractal_Volatility_Proximity'] / max_proximity.replace(0, np.nan)
    
    data['Raw_Fractal_Adaptive_Factor'] = (
        data['High_Volatility_Fractal_Composite'] * (1 - data['Fractal_Smooth_Transition']) +
        data['Low_Volatility_Fractal_Composite'] * data['Fractal_Smooth_Transition']
    )
    
    # Volume-Fractal Divergence Signals
    # Fractal Momentum Divergence
    data['Price_Fractal_Momentum'] = (
        (data['high'] - data['High_max_prev2']) * (data['close'] / data['close'].shift(1) - 1)
    )
    
    data['Volume_Fractal_Momentum'] = (
        (data['volume'] / data['volume'].shift(1) - 1) * data['Fractal_Volatility_Pattern']
    )
    
    data['Fractal_Momentum_Divergence'] = (
        data['Price_Fractal_Momentum'] - data['Volume_Fractal_Momentum']
    )
    
    # Efficiency-Volume Fractal Asymmetry
    data['High_Low_Volume_Distribution'] = (
        data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2))
    )
    
    data['Fractal_Amount_Efficiency'] = (
        data['amount'] / (data['volume'] * data['close'])
    ) * (data['volume'] / data['volume'].shift(1) - 1)
    
    data['Volume_Efficiency_Fractal_Asymmetry'] = (
        data['High_Low_Volume_Distribution'] * data['Fractal_Amount_Efficiency']
    )
    
    # Fractal Divergence Integration
    data['Primary_Fractal_Divergence'] = (
        data['Fractal_Momentum_Divergence'] * data['Efficiency_Volume_Divergence']
    )
    
    data['Secondary_Fractal_Divergence'] = (
        data['Volume_Efficiency_Fractal_Asymmetry'] * data['Fractal_Range_Efficiency']
    )
    
    data['Combined_Fractal_Divergence'] = (
        data['Primary_Fractal_Divergence'] + data['Secondary_Fractal_Divergence']
    )
    
    # Multi-Scale Fractal Signal Validation
    data['Fractal_Short_term_Validation'] = (
        data['Fractal_Price_Volume_Alignment'] * 
        (data['Fractal_Range_Efficiency'] / data['Fractal_Range_Efficiency'].shift(5) - 1)
    )
    
    data['Fractal_Medium_term_Alignment'] = (
        data['Volume_Fractal_Momentum_Cascade'] * data['Fractal_Volume_Efficiency']
    )
    
    fractal_range_median = data['Fractal_Range_Efficiency'].rolling(window=20, min_periods=1).median()
    data['Fractal_Long_term_Consistency'] = (
        (data['Fractal_Range_Efficiency'] > fractal_range_median).astype(int) * data['Fractal_Long_term_Trend']
    )
    
    # Fractal Liquidity Filtering
    data['Volume_Fractal_Concentration_Filter'] = (data['Volume_Fractal_Concentration'] > 1).astype(int)
    fractal_amount_median = data['Fractal_Amount_Efficiency'].rolling(window=20, min_periods=1).median()
    data['Fractal_Amount_Efficiency_Threshold'] = (data['Fractal_Amount_Efficiency'] > fractal_amount_median).astype(int)
    fractal_vol_pattern_median = data['Fractal_Volatility_Pattern'].rolling(window=20, min_periods=1).median()
    data['Fractal_Range_Constraint'] = (data['Fractal_Volatility_Pattern'] < fractal_vol_pattern_median).astype(int)
    
    # Final Fractal Alpha Output
    data['Fractal_Confirmation_Multiplier'] = (
        data['Fractal_Short_term_Validation'] * 
        data['Fractal_Medium_term_Alignment'] * 
        data['Fractal_Long_term_Consistency']
    )
    
    data['Fractal_Divergence_Enhancement'] = (
        data['Combined_Fractal_Divergence'] * data['Fractal_Confirmation_Multiplier']
    )
    
    # Apply fractal liquidity filtering conditions
    data['Fractal_Filter_Application'] = (
        data['Volume_Fractal_Concentration_Filter'] * 
        data['Fractal_Amount_Efficiency_Threshold'] * 
        data['Fractal_Range_Constraint']
    )
    
    # Final Fractal Alpha
    data['Final_Fractal_Alpha'] = (
        data['Raw_Fractal_Adaptive_Factor'] * 
        data['Fractal_Divergence_Enhancement'] * 
        data['Fractal_Filter_Application']
    )
    
    return data['Final_Fractal_Alpha']
