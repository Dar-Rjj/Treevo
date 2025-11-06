import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Multi-Scale Asymmetric Momentum
    # Micro Asymmetry
    micro_asym = ((data['high'] - data['open']) - (data['open'] - data['low'])) / (data['high'] - data['low'] + epsilon)
    
    # Meso Asymmetry
    high_2d = data['high'].rolling(window=3, min_periods=1).max()
    low_2d = data['low'].rolling(window=3, min_periods=1).min()
    meso_asym = ((high_2d - data['open']) - (data['open'] - low_2d)) / (high_2d - low_2d + epsilon)
    
    # Macro Asymmetry
    high_5d = data['high'].rolling(window=6, min_periods=1).max()
    low_5d = data['low'].rolling(window=6, min_periods=1).min()
    macro_asym = ((high_5d - data['open']) - (data['open'] - low_5d)) / (high_5d - low_5d + epsilon)
    
    # Combined Asymmetry
    combined_asym = (micro_asym + meso_asym + macro_asym) * (data['close'] - data['close'].shift(1)) / (data['close'].shift(4) - data['close'].shift(5) + epsilon)
    
    # Efficiency Regime Detection
    # Daily Efficiency
    daily_eff_denom = np.maximum(data['high'] - data['low'], 
                                np.maximum(np.abs(data['high'] - data['close'].shift(1)), 
                                          np.abs(data['low'] - data['close'].shift(1))))
    daily_efficiency = (data['close'] - data['open']) / (daily_eff_denom + epsilon)
    
    # Volatility Breakout
    vol_breakout = (data['high'] - data['low']) / (data['high'].shift(4) - data['low'].shift(4) + epsilon)
    
    # Regime Signal
    regime_signal = np.sign(daily_efficiency) * np.sign(vol_breakout - 1)
    
    # Volume-Price Dynamics
    # Volume Intensity
    vol_intensity = (data['volume'] / (data['volume'].rolling(window=3, min_periods=1).mean() + epsilon) - 
                    data['volume'] / (data['volume'].rolling(window=6, min_periods=1).mean() + epsilon))
    
    # Momentum Acceleration
    mom_accel = ((data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))) / (data['close'].shift(4) - data['close'].shift(5) + epsilon)
    
    # Acceleration Confirmation
    accel_confirmation = np.where(np.sign(vol_intensity) == np.sign(mom_accel), 
                                 np.abs(vol_intensity), -np.abs(vol_intensity))
    
    # Intraday Pressure Analysis
    # Morning Pressure
    morning_pressure = (data['high'] - data['open']) * data['amount'] - (data['open'] - data['low']) * data['amount']
    
    # Afternoon Pressure
    afternoon_pressure = (data['close'] - data['low']) * data['amount'] - (data['high'] - data['close']) * data['amount']
    
    # Pressure Asymmetry
    pressure_asymmetry = morning_pressure * afternoon_pressure * np.sign(morning_pressure - afternoon_pressure)
    
    # Adaptive Core Construction
    # High Volatility Core
    high_vol_core = combined_asym * np.abs(vol_breakout - 1) * accel_confirmation
    
    # Low Volatility Core
    low_vol_core = daily_efficiency * pressure_asymmetry
    
    # Regime-Adaptive Core
    regime_adaptive_core = np.where(vol_breakout > 1, high_vol_core, low_vol_core)
    
    # Quality Enhancement
    # Efficiency Consistency
    efficiency_consistency = daily_efficiency.rolling(window=6, min_periods=1).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) if len(x) > 0 else 0, raw=False
    )
    
    # Asymmetry Persistence
    def calc_asym_persistence(series):
        if len(series) < 3:
            return 0.0
        recent = series.iloc[-3:]
        micro_signs = np.sign(micro_asym.loc[recent.index])
        meso_signs = np.sign(meso_asym.loc[recent.index])
        matches = np.sum(micro_signs == meso_signs)
        return matches / 3.0
    
    asymmetry_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:
            window_data = data.iloc[:i+1]
            asymmetry_persistence.iloc[i] = calc_asym_persistence(window_data['close'])
        else:
            asymmetry_persistence.iloc[i] = 0.0
    
    # Quality Multiplier
    quality_multiplier = efficiency_consistency * asymmetry_persistence
    
    # Convergence Dynamics
    # Momentum-Efficiency Divergence
    mom_eff_divergence = combined_asym - daily_efficiency
    
    # Convergence Signal
    convergence_signal = np.sign(mom_eff_divergence) * regime_signal
    
    # Final Alpha Construction
    # Core Factor
    core_factor = regime_adaptive_core * quality_multiplier
    
    # Dynamic Enhancement
    dynamic_enhancement = core_factor * convergence_signal * accel_confirmation
    
    # Final Alpha
    final_alpha = dynamic_enhancement
    
    return final_alpha
