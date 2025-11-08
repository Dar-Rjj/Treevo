import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal-Gap Asymmetry Momentum with Multi-Regime Microstructure Dynamics
    """
    data = df.copy()
    
    # Calculate fractal levels (using 5-day window)
    data['Fractal_High'] = data['high'].rolling(window=5, min_periods=3).max().shift(1)
    data['Fractal_Low'] = data['low'].rolling(window=5, min_periods=3).min().shift(1)
    data['Fractal_Midpoint'] = (data['Fractal_High'] + data['Fractal_Low']) / 2
    
    # 1. Fractal Gap Asymmetry Analysis
    # Directional Fractal Gap Absorption
    upside_mask = (data['open'] > data['close'].shift(1)) & (data['open'] > data['Fractal_High'])
    downside_mask = (data['open'] < data['close'].shift(1)) & (data['open'] < data['Fractal_Low'])
    
    data['Upside_Fractal_Gap_Absorption'] = np.where(
        upside_mask,
        (data['high'] - data['open']) / (data['open'] - data['close'].shift(1)),
        0
    )
    data['Downside_Fractal_Gap_Absorption'] = np.where(
        downside_mask,
        (data['open'] - data['low']) / (data['close'].shift(1) - data['open']),
        0
    )
    data['Fractal_Gap_Absorption_Asymmetry'] = (
        data['Upside_Fractal_Gap_Absorption'] - data['Downside_Fractal_Gap_Absorption']
    )
    
    # Volume-Weighted Fractal Gap Efficiency
    vol_ratio = data['volume'] / data['volume'].shift(1)
    data['Volume_Upside_Fractal_Absorption'] = data['Upside_Fractal_Gap_Absorption'] * vol_ratio
    data['Volume_Downside_Fractal_Absorption'] = data['Downside_Fractal_Gap_Absorption'] * vol_ratio
    data['Volume_Fractal_Gap_Efficiency_Divergence'] = (
        data['Volume_Upside_Fractal_Absorption'] - data['Volume_Downside_Fractal_Absorption']
    )
    
    # Multi-Period Fractal Gap Dynamics
    data['Fractal_Gap_Absorption_Momentum'] = (
        data['Fractal_Gap_Absorption_Asymmetry'] - data['Fractal_Gap_Absorption_Asymmetry'].shift(3)
    )
    data['Volume_Fractal_Gap_Efficiency_Trend'] = (
        data['Volume_Fractal_Gap_Efficiency_Divergence'] / data['Volume_Fractal_Gap_Efficiency_Divergence'].shift(5) - 1
    )
    
    # 2. Volatility-Fractal Microstructure Integration
    # Fractal Intraday Volatility Asymmetry
    fractal_range = data['Fractal_High'] - data['Fractal_Low']
    data['Fractal_Opening_Volatility_Efficiency'] = (
        (data['high'] - data['open']) / fractal_range
    )
    data['Fractal_Closing_Volatility_Efficiency'] = (
        (data['close'] - data['low']) / fractal_range
    )
    data['Fractal_Intraday_Volatility_Skew'] = (
        data['Fractal_Opening_Volatility_Efficiency'] - data['Fractal_Closing_Volatility_Efficiency']
    )
    
    # Fractal Gap-Volatility Alignment
    data['Fractal_Gap_Opening_Volatility_Correlation'] = (
        np.sign(data['Fractal_Gap_Absorption_Asymmetry']) * np.sign(data['Fractal_Opening_Volatility_Efficiency'])
    )
    data['Fractal_Gap_Closing_Volatility_Correlation'] = (
        np.sign(data['Fractal_Gap_Absorption_Asymmetry']) * np.sign(data['Fractal_Closing_Volatility_Efficiency'])
    )
    data['Fractal_Volatility_Gap_Consistency'] = (
        data['Fractal_Gap_Opening_Volatility_Correlation'] + data['Fractal_Gap_Closing_Volatility_Correlation']
    )
    
    # 3. Fractal Volume Acceleration Asymmetry
    # Fractal Directional Volume Momentum
    upside_vol_mask = data['close'] > data['Fractal_Midpoint']
    downside_vol_mask = data['close'] < data['Fractal_Midpoint']
    
    vol_acc_3d = data['volume'] / data['volume'].shift(3) - 1
    data['Fractal_Upside_Volume_Acceleration'] = np.where(upside_vol_mask, vol_acc_3d, 0)
    data['Fractal_Downside_Volume_Acceleration'] = np.where(downside_vol_mask, vol_acc_3d, 0)
    data['Fractal_Volume_Acceleration_Asymmetry'] = (
        data['Fractal_Upside_Volume_Acceleration'] - data['Fractal_Downside_Volume_Acceleration']
    )
    
    # 4. Fractal Pressure Accumulation Dynamics
    # Fractal-Based Pressure Asymmetry
    fractal_upside_pressure = []
    fractal_downside_pressure = []
    
    for i in range(len(data)):
        if i < 4:
            fractal_upside_pressure.append(0)
            fractal_downside_pressure.append(0)
            continue
            
        upside_pressure = 0
        downside_pressure = 0
        for j in range(i-4, i+1):
            if data['close'].iloc[j] > data['Fractal_Midpoint'].iloc[j]:
                upside_pressure += data['volume'].iloc[j] * (data['close'].iloc[j] - data['Fractal_Midpoint'].iloc[j])
            elif data['close'].iloc[j] < data['Fractal_Midpoint'].iloc[j]:
                downside_pressure += data['volume'].iloc[j] * (data['Fractal_Midpoint'].iloc[j] - data['close'].iloc[j])
        
        fractal_upside_pressure.append(upside_pressure)
        fractal_downside_pressure.append(downside_pressure)
    
    data['Fractal_Upside_Pressure'] = fractal_upside_pressure
    data['Fractal_Downside_Pressure'] = fractal_downside_pressure
    data['Fractal_Pressure_Asymmetry'] = data['Fractal_Upside_Pressure'] - data['Fractal_Downside_Pressure']
    
    # 5. Composite Fractal Alpha Signal Generation
    # Core components with appropriate weighting
    gap_component = data['Volume_Fractal_Gap_Efficiency_Divergence'] * 0.3
    volatility_component = data['Fractal_Intraday_Volatility_Skew'] * 0.25
    volume_component = data['Fractal_Volume_Acceleration_Asymmetry'] * 0.25
    pressure_component = data['Fractal_Pressure_Asymmetry'] * 0.2
    
    # Regime-based adjustments
    strong_upside_absorption = (data['Fractal_Gap_Absorption_Asymmetry'] > 0.4) & (data['open'] > data['Fractal_High'])
    strong_downside_absorption = (data['Fractal_Gap_Absorption_Asymmetry'] < -0.4) & (data['open'] < data['Fractal_Low'])
    
    # Volume confirmation multiplier
    volume_confirmation = np.where(
        data['Fractal_Volume_Acceleration_Asymmetry'] * data['Fractal_Gap_Absorption_Asymmetry'] > 0,
        1.2, 0.8
    )
    
    # Volatility alignment multiplier
    volatility_alignment = np.where(
        data['Fractal_Volatility_Gap_Consistency'] > 0,
        1.1, 0.9
    )
    
    # Final composite signal
    base_signal = (
        gap_component + 
        volatility_component + 
        volume_component + 
        pressure_component
    )
    
    # Apply regime-based enhancements
    enhanced_signal = base_signal * volume_confirmation * volatility_alignment
    
    # Extreme regime detection for additional weighting
    extreme_upside = strong_upside_absorption & (data['Fractal_Volume_Acceleration_Asymmetry'] > 0.1)
    extreme_downside = strong_downside_absorption & (data['Fractal_Volume_Acceleration_Asymmetry'] < -0.1)
    
    final_signal = np.where(
        extreme_upside, enhanced_signal * 1.3,
        np.where(extreme_downside, enhanced_signal * 1.3, enhanced_signal)
    )
    
    return pd.Series(final_signal, index=data.index, name='fractal_alpha')
