import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Reversal-Volume Regime Factor
    Combines multi-horizon price reversal signals with dynamic volatility regimes
    and volume confirmation for enhanced return prediction
    """
    data = df.copy()
    
    # Multi-Horizon Price Reversal
    data['Rev_1d'] = -(data['close'] / data['close'].shift(1) - 1)
    data['Rev_3d'] = -(data['close'] / data['close'].shift(3) - 1)
    data['Rev_Accel'] = data['Rev_1d'] - (data['close'].shift(1) / data['close'].shift(2) - 1)
    
    # Composite Reversal Score
    data['Composite_Rev'] = 0.6 * data['Rev_1d'] + 0.3 * data['Rev_3d'] + 0.1 * data['Rev_Accel']
    data['Transformed_Rev'] = np.sign(data['Composite_Rev']) * np.sqrt(np.abs(data['Composite_Rev']))
    
    # Dynamic Volatility Regime
    data['Range_Vol'] = (data['high'] - data['low']) / data['close']
    data['Ret_Vol'] = (data['close'] / data['close'].shift(1) - 1).rolling(window=3).std()
    data['Vol_Composite'] = data['Range_Vol'] * data['Ret_Vol']
    
    # Volatility regime classification
    vol_threshold_high = data['Vol_Composite'].rolling(window=10).apply(lambda x: np.percentile(x, 80), raw=True)
    vol_threshold_medium = data['Vol_Composite'].rolling(window=10).apply(lambda x: np.percentile(x, 90), raw=True)
    
    data['Vol_Regime'] = 'Low'
    data.loc[data['Vol_Composite'] > vol_threshold_medium, 'Vol_Regime'] = 'Medium'
    data.loc[data['Vol_Composite'] > vol_threshold_high, 'Vol_Regime'] = 'High'
    
    # Volume Confirmation System
    data['Vol_Mom'] = data['volume'] / data['volume'].shift(1) - 1
    data['Vol_Trend'] = data['volume'] / data['volume'].shift(3) - 1
    data['Vol_Stability'] = 1 / (np.abs(data['Vol_Mom']) + 1e-6)
    
    data['Dir_Align'] = np.sign(data['Rev_1d']) * np.sign(data['Vol_Mom'])
    data['Str_Align'] = np.abs(data['Rev_1d']) / (np.abs(data['Vol_Mom']) + 1e-6)
    data['Vol_Score'] = data['Dir_Align'] * np.log(1 + np.abs(data['Str_Align']))
    
    # Regime-Adaptive Factor Construction
    # Volatility regime multipliers
    data['Vol_Regime_Mult'] = 1.0
    data.loc[data['Vol_Regime'] == 'High', 'Vol_Regime_Mult'] = 1.8
    data.loc[data['Vol_Regime'] == 'Medium', 'Vol_Regime_Mult'] = 1.2
    data.loc[data['Vol_Regime'] == 'Low', 'Vol_Regime_Mult'] = 0.7
    
    # Volume confirmation multipliers
    data['Vol_Conf_Mult'] = 0.8  # Default: Weak/Contrary
    data.loc[(data['Vol_Score'] > 0) & (data['Vol_Trend'] > 0), 'Vol_Conf_Mult'] = 1.1  # Moderate
    data.loc[(data['Vol_Score'] > 0.3) & (data['Vol_Trend'] > 0.15), 'Vol_Conf_Mult'] = 1.5  # Strong
    
    # Volume stability multipliers
    stab_threshold_high = data['Vol_Stability'].rolling(window=5).apply(lambda x: np.percentile(x, 85), raw=True)
    stab_threshold_medium = data['Vol_Stability'].rolling(window=5).apply(lambda x: np.percentile(x, 95), raw=True)
    
    data['Vol_Stab_Mult'] = 0.9  # Low stability
    data.loc[data['Vol_Stability'] > stab_threshold_medium, 'Vol_Stab_Mult'] = 1.0  # Medium
    data.loc[data['Vol_Stability'] > stab_threshold_high, 'Vol_Stab_Mult'] = 1.3  # High
    
    # Final Alpha Integration
    data['Base_Factor'] = (data['Transformed_Rev'] * 
                          data['Vol_Regime_Mult'] * 
                          data['Vol_Conf_Mult'] * 
                          data['Vol_Stab_Mult'])
    
    data['Adjusted_Factor'] = data['Base_Factor'] / (data['Range_Vol'] + 1e-6)
    
    return data['Adjusted_Factor']
