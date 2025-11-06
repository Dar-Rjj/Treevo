import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Momentum Asymmetry
    data['Opening_Fractal_Strength'] = (data['open'] - data['low']) / (data['high'] - data['low'] + 0.001)
    data['Closing_Fractal_Strength'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 0.001)
    data['Momentum_Transmission_Divergence'] = (data['high'] - data['open']) * data['Opening_Fractal_Strength'] - (data['close'] - data['low']) * data['Closing_Fractal_Strength']
    
    # Calculate consecutive days for Fractal Persistence Divergence
    data['Opening_Strength_Above_05'] = data['Opening_Fractal_Strength'] > 0.5
    data['Closing_Strength_Above_05'] = data['Closing_Fractal_Strength'] > 0.5
    
    opening_streak = data['Opening_Strength_Above_05'].astype(int)
    closing_streak = data['Closing_Strength_Above_05'].astype(int)
    
    for i in range(1, len(data)):
        if data['Opening_Strength_Above_05'].iloc[i]:
            opening_streak.iloc[i] = opening_streak.iloc[i-1] + 1
        else:
            opening_streak.iloc[i] = 0
            
        if data['Closing_Strength_Above_05'].iloc[i]:
            closing_streak.iloc[i] = closing_streak.iloc[i-1] + 1
        else:
            closing_streak.iloc[i] = 0
    
    data['Fractal_Persistence_Divergence'] = opening_streak - closing_streak
    
    # Multi-Scale Momentum Acceleration
    data['Micro_Acceleration'] = ((data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))) / (abs(data['close'].shift(1) - data['close'].shift(2)) + 0.001) * np.sign(data['Momentum_Transmission_Divergence'])
    data['Meso_Acceleration'] = ((data['close'] - data['close'].shift(5)) - (data['close'].shift(5) - data['close'].shift(10))) / (abs(data['close'].shift(5) - data['close'].shift(10)) + 0.001) * np.sign(data['Fractal_Persistence_Divergence'])
    
    macro_numerator = (data['high'] - data['open']) * data['Opening_Fractal_Strength'] / ((data['close'] - data['low']) * data['Closing_Fractal_Strength'] + 0.001) - 1
    data['Macro_Acceleration'] = ((data['close'] - data['close'].shift(13)) - (data['close'].shift(13) - data['close'].shift(26))) / (abs(data['close'].shift(13) - data['close'].shift(26)) + 0.001) * np.sign(macro_numerator)
    
    # Volume-Momentum Fractal Transmission
    data['Volume_Fractal_Divergence'] = data['volume'] * data['Opening_Fractal_Strength'] - data['volume'] * data['Closing_Fractal_Strength']
    data['Pressure_Imbalance'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 0.001) - (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)
    
    # Calculate Volume Transmission Stability
    volume_div_sign_change = (data['Volume_Fractal_Divergence'] * data['Volume_Fractal_Divergence'].shift(1) < 0).cumsum()
    volume_div_pos_streak = (data['Volume_Fractal_Divergence'] > 0).astype(int)
    
    for i in range(1, len(data)):
        if data['Volume_Fractal_Divergence'].iloc[i] > 0:
            volume_div_pos_streak.iloc[i] = volume_div_pos_streak.iloc[i-1] + 1
        else:
            volume_div_pos_streak.iloc[i] = 0
    
    data['Volume_Transmission_Stability'] = volume_div_pos_streak / (volume_div_sign_change.diff().fillna(0).cumsum() + 1)
    
    # Momentum Regime Integration
    regime_condition = (data['high'] - data['open']) * data['Opening_Fractal_Strength'] / ((data['close'] - data['low']) * data['Closing_Fractal_Strength'] + 0.001)
    
    data['Regime_Signal'] = 0.0
    
    # Regime 1
    mask1 = (regime_condition > 1.2) & (data['Momentum_Transmission_Divergence'] > 0)
    volume_ma = (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3) + data['volume'].shift(4)) / 4
    price_move_current = abs((data['close'] - data['open']) / data['open'])
    price_move_prev = abs((data['open'] - data['close'].shift(1)) / data['close'].shift(1))
    data.loc[mask1, 'Regime_Signal'] = (data['volume'] / (volume_ma + 0.001) * 
                                       price_move_current / (price_move_current + price_move_prev + 0.001) * 
                                       data['Pressure_Imbalance'])
    
    # Regime 2
    mask2 = (regime_condition > 1.2) & (data['Momentum_Transmission_Divergence'] <= 0)
    high_max = data['high'].rolling(window=4, min_periods=1).max().shift(1)
    data.loc[mask2, 'Regime_Signal'] = (data['volume'] / data['volume'].shift(3) * 
                                       (data['high'] - high_max) / data['close'].shift(1) * 
                                       data['Volume_Fractal_Divergence'])
    
    # Regime 3
    mask3 = (regime_condition <= 1.2) & (data['Momentum_Transmission_Divergence'] > 0)
    volume_sum = (data['volume'] + data['volume'].shift(1) + data['volume'].shift(2) + 
                 data['volume'].shift(3) + data['volume'].shift(4))
    data.loc[mask3, 'Regime_Signal'] = (data['volume'] / (volume_sum + 0.001) * 
                                       (1 - (data['high'] - data['low']) / data['close']) * 
                                       data['volume'] / (volume_sum + 0.001) * 
                                       data['Pressure_Imbalance'])
    
    # Regime 4
    mask4 = (regime_condition <= 1.2) & (data['Momentum_Transmission_Divergence'] <= 0)
    data.loc[mask4, 'Regime_Signal'] = ((data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 0.001) * 
                                       data['volume'] / (volume_sum + 0.001) * 
                                       abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 0.001) * 
                                       data['Volume_Transmission_Stability'])
    
    # Gap Momentum Transmission
    data['Overnight_Gap'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 0.001)
    data['Intraday_Recovery'] = (data['close'] - data['open']) / (abs(data['open'] - data['close'].shift(1)) + 0.001)
    data['Gap_Momentum_Transmission'] = (data['Overnight_Gap'] * data['Intraday_Recovery'] * 
                                        (data['Micro_Acceleration'] + data['Meso_Acceleration'] + data['Macro_Acceleration']))
    
    # Fractal Transmission Quality
    momentum_sign_change = (data['Momentum_Transmission_Divergence'] * data['Momentum_Transmission_Divergence'].shift(1) < 0).cumsum()
    momentum_pos_streak = (data['Momentum_Transmission_Divergence'] > 0).astype(int)
    
    for i in range(1, len(data)):
        if data['Momentum_Transmission_Divergence'].iloc[i] > 0:
            momentum_pos_streak.iloc[i] = momentum_pos_streak.iloc[i-1] + 1
        else:
            momentum_pos_streak.iloc[i] = 0
    
    data['Fractal_Pattern_Quality'] = momentum_pos_streak * (momentum_pos_streak / (momentum_sign_change.diff().fillna(0).cumsum() + 1))
    
    volume_div_sign_change = (data['Volume_Fractal_Divergence'] * data['Volume_Fractal_Divergence'].shift(1) < 0).cumsum()
    volume_div_pos_streak = (data['Volume_Fractal_Divergence'] > 0).astype(int)
    
    for i in range(1, len(data)):
        if data['Volume_Fractal_Divergence'].iloc[i] > 0:
            volume_div_pos_streak.iloc[i] = volume_div_pos_streak.iloc[i-1] + 1
        else:
            volume_div_pos_streak.iloc[i] = 0
    
    data['Volume_Transmission_Quality'] = volume_div_pos_streak * (volume_div_pos_streak / (volume_div_sign_change.diff().fillna(0).cumsum() + 1))
    
    # Final Alpha Construction
    data['Core_Momentum_Signal'] = (data['Micro_Acceleration'] + data['Meso_Acceleration'] + data['Macro_Acceleration']) * data['Regime_Signal']
    data['Enhanced_Momentum_Signal'] = data['Core_Momentum_Signal'] * data['Gap_Momentum_Transmission'] * data['Volume_Transmission_Stability']
    data['Quality_Enhanced_Momentum_Alpha'] = (data['Enhanced_Momentum_Signal'] * data['Pressure_Imbalance'] * 
                                             data['Fractal_Pattern_Quality'] * data['Volume_Transmission_Quality'])
    
    return data['Quality_Enhanced_Momentum_Alpha']
