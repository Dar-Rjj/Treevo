import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Fracture Framework
    # Intraday Fracture
    intraday_fracture = (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-6) * (data['open'] - data['close'].shift(2))
    
    # Gap Fracture
    gap_fracture = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)) * np.abs(data['close'] - data['open'])
    
    # Price Fracture Divergence
    price_fracture_divergence = intraday_fracture - gap_fracture * np.sign(data['close'] - data['close'].shift(2))
    
    # Volume Fracture Framework
    # Volume Intensity
    volume_intensity = data['volume'] / data['volume'].shift(3) * np.abs(data['close'] - data['close'].shift(1))
    
    # Volume Persistence
    volume_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            count_volume_increase = sum(window['volume'].iloc[j] > window['volume'].iloc[j-1] for j in range(1, 5))
            volume_persistence.iloc[i] = count_volume_increase * data['volume'].iloc[i] / window['volume'].sum()
        else:
            volume_persistence.iloc[i] = np.nan
    
    # Volume Fracture Divergence
    volume_fracture_divergence = volume_intensity - volume_persistence * np.sign(data['close'] - data['open'])
    
    # Multi-Timeframe Integration
    # Short-Term Fracture
    short_term_fracture = (data['close'] - data['open']) * (data['high'] - data['close'].shift(1)) / (np.abs(data['close'] - data['close'].shift(1)) + 1e-6)
    
    # Medium-Term Fracture
    medium_term_fracture = (data['close'] - data['close'].shift(5)) * np.abs(data['close'] - data['close'].shift(3))
    
    # Timeframe Divergence
    timeframe_divergence = short_term_fracture - medium_term_fracture
    
    # Amount Fracture Framework
    # Amount Intensity
    amount_intensity = data['amount'] / data['amount'].shift(3) * (data['close'] - data['open'])
    
    # Amount Persistence
    amount_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            count_amount_increase = sum(window['amount'].iloc[j] > window['amount'].iloc[j-1] for j in range(1, 5))
            amount_persistence.iloc[i] = count_amount_increase * data['amount'].iloc[i] / window['amount'].sum()
        else:
            amount_persistence.iloc[i] = np.nan
    
    # Amount Fracture
    amount_fracture = amount_intensity - amount_persistence * np.sign(data['close'] - data['close'].shift(2))
    
    # Fracture Integration
    # Triple Fracture Composite
    triple_fracture_composite = price_fracture_divergence * volume_fracture_divergence * amount_fracture
    
    # Fracture Momentum
    fracture_momentum = triple_fracture_composite * (data['close'] - data['close'].shift(2))
    
    # Fracture Persistence
    fracture_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 7:
            window = triple_fracture_composite.iloc[i-7:i]
            current_sign = np.sign(triple_fracture_composite.iloc[i])
            persistence_count = sum(np.sign(window.iloc[j]) == current_sign for j in range(len(window)))
            fracture_persistence.iloc[i] = persistence_count
        else:
            fracture_persistence.iloc[i] = np.nan
    
    # Final Alpha Construction
    # Core Fracture Alpha
    core_fracture_alpha = fracture_momentum * fracture_persistence * timeframe_divergence
    
    # Final Alpha
    final_alpha = core_fracture_alpha * np.sign(data['close'] - data['close'].shift(1))
    
    return final_alpha
