import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Core Reversal Signals
    data['Rev_1d'] = (data['close'].shift(1) - data['close']) / data['close']
    data['Rev_3d'] = (data['close'].shift(3) - data['close']) / data['close']
    data['Daily_Range_Vol'] = (data['high'] - data['low']) / data['close']
    data['Vol_Adj_Rev'] = (data['Rev_1d'] + data['Rev_3d']) / (data['Daily_Range_Vol'] + 1e-6)
    
    # Volatility Regime Classification
    data['Range_Vol'] = (data['high'] - data['low']) / data['close']
    data['Return_Vol'] = abs(data['close'] / data['close'].shift(1) - 1)
    data['Composite_Vol'] = (data['Range_Vol'] + data['Return_Vol']) / 2
    
    # Volume Confirmation Framework
    data['Vol_Trend_1d'] = data['volume'] / data['volume'].shift(1) - 1
    data['Vol_Trend_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['Vol_Accel'] = data['Vol_Trend_1d'] - data['Vol_Trend_3d']
    
    # Initialize signals and multipliers
    data['Base_Signal'] = 0.0
    data['Volume_Multiplier'] = 1.0
    
    # Regime-Adaptive Signal Construction
    for i in range(len(data)):
        if pd.isna(data['Composite_Vol'].iloc[i]) or pd.isna(data['Vol_Accel'].iloc[i]):
            continue
            
        # Volatility regime classification
        comp_vol = data['Composite_Vol'].iloc[i]
        if comp_vol > 0.02:
            # High Vol: Focus on short-term reversal
            data.loc[data.index[i], 'Base_Signal'] = data['Rev_1d'].iloc[i]
        elif comp_vol > 0.01:
            # Medium Vol: Balanced approach
            data.loc[data.index[i], 'Base_Signal'] = 0.6 * data['Rev_1d'].iloc[i] + 0.4 * data['Rev_3d'].iloc[i]
        else:
            # Low Vol: Emphasize volatility adjustment
            data.loc[data.index[i], 'Base_Signal'] = data['Vol_Adj_Rev'].iloc[i]
        
        # Volume confirmation multipliers
        vol_accel = data['Vol_Accel'].iloc[i]
        vol_trend_3d = data['Vol_Trend_3d'].iloc[i]
        
        if vol_accel > 0.1 and vol_trend_3d > 0.2:
            data.loc[data.index[i], 'Volume_Multiplier'] = 1.5
        elif vol_accel > 0 and vol_trend_3d > 0:
            data.loc[data.index[i], 'Volume_Multiplier'] = 1.1
        else:
            data.loc[data.index[i], 'Volume_Multiplier'] = 0.7
    
    # Volume adjusted signal
    data['Volume_Adjusted_Signal'] = data['Base_Signal'] * data['Volume_Multiplier']
    
    # Final Factor Enhancement
    data['Vol_Scaled'] = data['Volume_Adjusted_Signal'] / (data['Composite_Vol'] + 1e-6)
    data['Amount_Enhanced'] = data['Vol_Scaled'] * np.log(data['amount'] + 1)
    
    # Return the final factor
    return data['Amount_Enhanced']
