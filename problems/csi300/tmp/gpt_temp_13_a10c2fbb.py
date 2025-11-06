import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Fractal Momentum
    # Micro-Fractal (3-day)
    micro_fractal = (data['close'] - data['close'].shift(3)) / (
        data['high'].rolling(window=3, min_periods=3).max() - 
        data['low'].rolling(window=3, min_periods=3).min()
    )
    
    # Meso-Fractal (8-day)
    meso_fractal = (data['close'] - data['close'].shift(8)) / (
        data['high'].rolling(window=8, min_periods=8).max() - 
        data['low'].rolling(window=8, min_periods=8).min()
    )
    
    # Macro-Fractal (21-day)
    macro_fractal = (data['close'] - data['close'].shift(21)) / (
        data['high'].rolling(window=21, min_periods=21).max() - 
        data['low'].rolling(window=21, min_periods=21).min()
    )
    
    # Fractal Momentum Regimes
    fractal_convergence = (micro_fractal > meso_fractal) & (meso_fractal > macro_fractal)
    fractal_divergence = (micro_fractal < meso_fractal) & (meso_fractal < macro_fractal)
    fractal_chaos = ~(fractal_convergence | fractal_divergence)
    
    # Pressure-Entropy Dynamics
    buy_pressure_intensity = data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low'])
    sell_pressure_intensity = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'])
    pressure_entropy_ratio = buy_pressure_intensity / sell_pressure_intensity
    
    # Pressure-Entropy Coupling
    high_pressure_entropy = fractal_convergence & (pressure_entropy_ratio > 1.2)
    low_pressure_entropy = fractal_divergence & (pressure_entropy_ratio < 0.8)
    neutral_pressure_entropy = ~(high_pressure_entropy | low_pressure_entropy)
    
    coupling_strength = pd.Series(0.0, index=data.index)
    coupling_strength[high_pressure_entropy] = buy_pressure_intensity[high_pressure_entropy] * micro_fractal[high_pressure_entropy]
    coupling_strength[low_pressure_entropy] = sell_pressure_intensity[low_pressure_entropy] * macro_fractal[low_pressure_entropy]
    
    # Microstructure Momentum Patterns
    opening_momentum_persistence = (data['close'] - data['open']) * (data['open'] - data['close'].shift(1))
    midday_reversal_intensity = (data['high'] + data['low']) / 2 - (data['open'] + data['close']) / 2
    volume_price_efficiency = (data['close'] - data['close'].shift(1)) * data['volume'] / data['amount']
    
    # Fractal-Microstructure Interference
    constructive_microstructure = high_pressure_entropy & (opening_momentum_persistence > 0)
    destructive_microstructure = high_pressure_entropy & (opening_momentum_persistence < 0)
    neutral_microstructure = ~(constructive_microstructure | destructive_microstructure)
    
    microstructure_power = pd.Series(0.0, index=data.index)
    
    # Constructive microstructure persistence
    constructive_persistence = pd.Series(0, index=data.index, dtype=int)
    for i in range(1, len(data)):
        if constructive_microstructure.iloc[i]:
            constructive_persistence.iloc[i] = constructive_persistence.iloc[i-1] + 1
    
    # Destructive microstructure persistence  
    destructive_persistence = pd.Series(0, index=data.index, dtype=int)
    for i in range(1, len(data)):
        if destructive_microstructure.iloc[i]:
            destructive_persistence.iloc[i] = destructive_persistence.iloc[i-1] + 1
    
    microstructure_power[constructive_microstructure] = (
        coupling_strength[constructive_microstructure] * 
        volume_price_efficiency[constructive_microstructure]
    )
    microstructure_power[destructive_microstructure] = (
        abs(coupling_strength[destructive_microstructure]) * 
        abs(midday_reversal_intensity[destructive_microstructure])
    )
    
    # Trade Size Fractal Dynamics
    avg_trade_size = data['amount'] / data['volume']
    
    # Large trade flow (top 20%)
    large_trade_threshold = avg_trade_size.rolling(window=5, min_periods=5).apply(
        lambda x: np.percentile(x, 80), raw=True
    )
    large_trade_flow = avg_trade_size.where(avg_trade_size > large_trade_threshold)
    
    # Small trade flow (bottom 20%)
    small_trade_threshold = avg_trade_size.rolling(window=5, min_periods=5).apply(
        lambda x: np.percentile(x, 20), raw=True
    )
    small_trade_flow = avg_trade_size.where(avg_trade_size < small_trade_threshold)
    
    # Large trade fractal dimension
    large_trade_fractal = pd.Series(np.nan, index=data.index)
    for i in range(4, len(data)):
        window = large_trade_flow.iloc[i-4:i+1]
        if window.notna().all():
            range_val = window.max() - window.min()
            if range_val > 0:
                sum_abs_diff = sum(abs(window.iloc[j] - window.iloc[j-1]) for j in range(1, len(window)))
                large_trade_fractal.iloc[i] = np.log(5) / np.log(sum_abs_diff / range_val)
    
    # Small trade fractal dimension
    small_trade_fractal = pd.Series(np.nan, index=data.index)
    for i in range(4, len(data)):
        window = small_trade_flow.iloc[i-4:i+1]
        if window.notna().all():
            range_val = window.max() - window.min()
            if range_val > 0:
                sum_abs_diff = sum(abs(window.iloc[j] - window.iloc[j-1]) for j in range(1, len(window)))
                small_trade_fractal.iloc[i] = np.log(5) / np.log(sum_abs_diff / range_val)
    
    size_fractal_ratio = large_trade_fractal / small_trade_fractal
    
    # Final Alpha Factor Synthesis
    # Base Fractal Microstructure
    base_factor = buy_pressure_intensity * volume_price_efficiency
    
    # Pressure-Entropy Modulation
    pressure_modulation = pd.Series(1.0, index=data.index)
    pressure_modulation[high_pressure_entropy] = 1 + coupling_strength[high_pressure_entropy]
    pressure_modulation[low_pressure_entropy] = 1 - coupling_strength[low_pressure_entropy]
    
    # Microstructure Interference Factor
    microstructure_interference = pd.Series(1.0, index=data.index)
    microstructure_interference[constructive_microstructure] = (
        1 + microstructure_power[constructive_microstructure] * constructive_persistence[constructive_microstructure]
    )
    microstructure_interference[destructive_microstructure] = (
        1 - microstructure_power[destructive_microstructure] * destructive_persistence[destructive_microstructure]
    )
    
    # Size Fractal Alignment
    size_pressure_convergence = size_fractal_ratio * pressure_entropy_ratio
    
    # Final Factor Assembly
    final_factor = (
        base_factor * 
        pressure_modulation * 
        microstructure_interference * 
        size_pressure_convergence * 
        pressure_entropy_ratio * 
        micro_fractal
    )
    
    return final_factor
