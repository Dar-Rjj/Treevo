import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Asymmetric Volatility Components
    data['Daily_Range_Asymmetry'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001) - 0.5
    data['Volatility_Skew'] = ((data['high'] - data['open']) / (data['high'] - data['low'] + 0.001) - 
                              (data['open'] - data['low']) / (data['high'] - data['low'] + 0.001))
    data['Movement_Efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    
    # Multi-Timeframe Volatility Dynamics
    data['Short_term_Volatility_Momentum'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5) + 0.001) - 1
    data['Medium_term_Volatility_Momentum'] = (data['high'] - data['low']) / (data['high'].shift(15) - data['low'].shift(15) + 0.001) - 1
    data['Volatility_Momentum_Divergence'] = abs(data['Short_term_Volatility_Momentum'] - data['Medium_term_Volatility_Momentum'])
    
    # Volume-Pressure Integration
    # Pressure Accumulation (5-day window)
    pressure_accum = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window_sum = 0
            for j in range(5):
                idx = i - j
                if data['high'].iloc[idx] != data['low'].iloc[idx]:
                    window_sum += (data['close'].iloc[idx] - data['open'].iloc[idx]) / (data['high'].iloc[idx] - data['low'].iloc[idx] + 0.001)
            pressure_accum.iloc[i] = window_sum
    data['Pressure_Accumulation'] = pressure_accum
    
    data['Volume_to_Volatility_Ratio'] = data['volume'] / ((data['high'] - data['low']) * data['close'] + 0.001)
    
    volume_change = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 0.001)
    price_change = (data['close'] - data['close'].shift(3)) / (data['close'].shift(3) + 0.001)
    data['Volume_Price_Alignment'] = np.sign(volume_change) * np.sign(price_change)
    
    # Asymmetric Fracture Detection
    volatility_acceleration = abs(data['Short_term_Volatility_Momentum'] - data['Medium_term_Volatility_Momentum'])
    medium_term_abs = abs(data['Medium_term_Volatility_Momentum'])
    data['Volatility_Acceleration_Fracture'] = (volatility_acceleration / (medium_term_abs + 0.001)) > 1.0
    
    # Calculate rolling max/min for momentum asymmetric fracture
    rolling_high = data['high'].rolling(window=6, min_periods=6).max()
    rolling_low = data['low'].rolling(window=6, min_periods=6).min()
    momentum_ratio = (data['close'] - data['close'].shift(5)) / (rolling_high - rolling_low + 0.001)
    momentum_asymmetry_diff = abs(data['Daily_Range_Asymmetry'] - momentum_ratio)
    momentum_abs = abs(momentum_ratio)
    data['Momentum_Asymmetric_Fracture'] = (momentum_asymmetry_diff / (momentum_abs + 0.001)) > 1.0
    
    data['Cross_Asymmetric_Fracture'] = data['Volatility_Acceleration_Fracture'] & data['Momentum_Asymmetric_Fracture']
    
    # Volume-Momentum Conformation
    # Directional Imbalance (5-day window)
    directional_imbalance = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            up_sum = 0
            down_sum = 0
            for j in range(5):
                idx = i - j
                if idx > 0:
                    price_diff = data['close'].iloc[idx] - data['close'].iloc[idx-1]
                    up_sum += max(0, price_diff)
                    down_sum += max(0, -price_diff)
            directional_imbalance.iloc[i] = (up_sum - down_sum) / (up_sum + down_sum + 0.001)
    data['Directional_Imbalance'] = directional_imbalance
    
    # Volume Surge
    data['Volume_Surge'] = data['volume'] / (data['volume'].rolling(window=5, min_periods=5).mean().shift(1) + 0.001)
    
    # Volume Asymmetry Divergence
    mid_price = (data['high'] + data['low']) / 2
    volume_above = data['volume'].where(data['close'] > mid_price, 0)
    volume_below = data['volume'].where(data['close'] < mid_price, 0)
    data['Volume_Asymmetry_Divergence'] = (volume_above / (volume_below + 0.001)) * np.sign(data['Daily_Range_Asymmetry'])
    
    # Multi-Scale Signal Integration
    data['Short_term_Signal'] = (data['Volatility_Skew'] * data['Movement_Efficiency'] * 
                                data['Daily_Range_Asymmetry'] * data['Volume_Price_Alignment'] * 
                                data['Pressure_Accumulation'] * data['Volume_to_Volatility_Ratio'])
    
    data['Medium_term_Signal'] = (data['Volatility_Momentum_Divergence'] * 
                                 data['Momentum_Asymmetric_Fracture'].astype(float) * 
                                 data['Directional_Imbalance'] * data['Volume_Asymmetry_Divergence'] * 
                                 data['Volume_Surge'] * data['Cross_Asymmetric_Fracture'].astype(float))
    
    data['Signal_Alignment'] = np.sign(data['Short_term_Signal']) * np.sign(data['Medium_term_Signal'])
    
    # Adaptive Alpha Construction
    alpha = pd.Series(index=data.index, dtype=float)
    
    high_vol_regime = data['Volatility_Momentum_Divergence'] > 0.1
    medium_vol_regime = (data['Volatility_Momentum_Divergence'] >= 0.05) & (data['Volatility_Momentum_Divergence'] <= 0.1)
    low_vol_regime = data['Volatility_Momentum_Divergence'] < 0.05
    
    # High Volatility Regime
    alpha[high_vol_regime] = (data['Cross_Asymmetric_Fracture'][high_vol_regime].astype(float) * 
                             data['Short_term_Signal'][high_vol_regime] * 
                             data['Medium_term_Signal'][high_vol_regime] * 
                             data['Signal_Alignment'][high_vol_regime] * 
                             data['Pressure_Accumulation'][high_vol_regime])
    
    # Medium Volatility Regime
    alpha[medium_vol_regime] = (data['Cross_Asymmetric_Fracture'][medium_vol_regime].astype(float) * 
                               data['Volatility_Momentum_Divergence'][medium_vol_regime] * 
                               data['Movement_Efficiency'][medium_vol_regime] * 
                               data['Volume_Price_Alignment'][medium_vol_regime] * 
                               data['Signal_Alignment'][medium_vol_regime])
    
    # Low Volatility Regime
    alpha[low_vol_regime] = (data['Cross_Asymmetric_Fracture'][low_vol_regime].astype(float) * 
                            data['Volume_Asymmetry_Divergence'][low_vol_regime] * 
                            data['Directional_Imbalance'][low_vol_regime] * 
                            data['Volume_Surge'][low_vol_regime] * 
                            data['Signal_Alignment'][low_vol_regime])
    
    return alpha
