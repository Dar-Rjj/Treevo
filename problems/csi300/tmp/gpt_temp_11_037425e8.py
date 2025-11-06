import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Remove first row with NaN values
    data = data.iloc[1:].copy()
    
    # Short-Term Volatility Fracture
    opening_vol_momentum = ((data['open'] - data['prev_close']) / (data['high'] - data['low'])) * \
                          (abs(data['open'] - data['prev_close']) / data['prev_close']) * \
                          ((data['close'] - data['prev_close']) / (data['high'] - data['low']))
    
    midday_vol_efficiency = ((data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])) * \
                           (data['volume'] / data['prev_volume']) * \
                           (abs(data['open'] - data['prev_close']) / data['prev_close']) * \
                           (abs(data['close'] - data['open']) / (data['high'] - data['low']))
    
    closing_vol_momentum = ((data['close'] - data['open']) / (data['high'] - data['low'])) * \
                          (abs(data['close'] - data['prev_close']) / data['prev_close']) * \
                          (data['volume'] / data['prev_volume']) * \
                          ((data['close'] - data['prev_close']) / (data['high'] - data['low']))
    
    short_term_vol_fracture = (opening_vol_momentum + midday_vol_efficiency + closing_vol_momentum) / 3
    
    # Medium-Term Volatility Persistence
    vol_range_expansion = ((data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))) * \
                         (data['volume'] / data['prev_volume']) * \
                         (abs(data['open'] - data['prev_close']) / (data['prev_high'] - data['prev_low']))
    
    # Volatility Momentum Continuity
    vol_momentum_continuity = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window = data.iloc[i-5:i]
        count = sum((np.sign(window['close'].diff()) == np.sign(window['close'].diff().shift(1))).dropna())
        vol_momentum_continuity.iloc[i] = count * ((data['close'].iloc[i] - data['close'].iloc[i-5]) / data['close'].iloc[i-5]) * \
                                        (abs(data['open'].iloc[i] - data['prev_close'].iloc[i]) / data['prev_close'].iloc[i])
    
    vol_price_efficiency = (abs(data['close'] - data['prev_close']) / (data['high'] - data['low'])) * \
                          (data['amount'] / data['prev_amount']) * \
                          (abs(data['open'] - data['prev_close']) / data['prev_close']) * \
                          ((data['close'] - data['close'].shift(5)) / (data['high'] - data['low']))
    
    medium_term_vol_persistence = (vol_range_expansion + vol_momentum_continuity + vol_price_efficiency) / 3
    
    # Long-Term Volatility Acceleration
    vol_speed = ((data['close'] - data['close'].shift(2)) / (data['close'].shift(1) - data['close'].shift(3))) * \
                (data['volume'] / data['prev_volume']) * \
                (abs(data['open'] - data['prev_close']) / (data['high'] - data['low'])) * \
                ((data['high'] - data['low']) / (data['high'].shift(20) - data['low'].shift(20)))
    
    # Volatility Stability
    vol_stability = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = data.iloc[i-4:i+1]
        std_close_diff = window['close'].diff().std()
        vol_stability.iloc[i] = (std_close_diff / abs(data['close'].iloc[i] - data['prev_close'].iloc[i])) * \
                               (data['volume'].iloc[i] / data['prev_volume'].iloc[i]) * \
                               (abs(data['open'].iloc[i] - data['prev_close'].iloc[i]) / data['prev_close'].iloc[i]) * \
                               ((data['close'].iloc[i] - data['close'].iloc[i-20]) / (data['high'].iloc[i] - data['low'].iloc[i]))
    
    # Volatility Convergence
    vol_convergence = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = data.iloc[i-4:i+1]
        avg_close_diff = window['close'].diff().abs().mean()
        vol_convergence.iloc[i] = ((data['close'].iloc[i] - data['prev_close'].iloc[i]) / avg_close_diff) * \
                                 (data['amount'].iloc[i] / data['prev_amount'].iloc[i]) * \
                                 (abs(data['open'].iloc[i] - data['prev_close'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i])) * \
                                 ((data['high'].iloc[i] - data['low'].iloc[i]) / (data['high'].iloc[i-20] - data['low'].iloc[i-20]))
    
    long_term_vol_acceleration = (vol_speed + vol_stability + vol_convergence) / 3
    
    # Multi-Scale Component
    multi_scale_component = (short_term_vol_fracture + medium_term_vol_persistence + long_term_vol_acceleration) / 3 * \
                           (data['volume'] / data['prev_volume']) * \
                           (abs(data['open'] - data['prev_close']) / data['prev_close'])
    
    # Volume-Volatility Component (simplified)
    vol_confirmation_strength = np.sign(data['close'] - data['prev_close']) * \
                               ((data['volume'] - data['prev_volume']) / data['prev_volume']) * \
                               (abs(data['close'] - data['prev_close']) / data['prev_close']) * \
                               (abs(data['open'] - data['prev_close']) / data['prev_close']) * \
                               (data['close'] - data['prev_close']) * \
                               (data['volume'] / data['amount']) * \
                               ((data['high'] - data['low']) / data['prev_close'])
    
    vol_amount_efficiency = (data['amount'] / data['volume']) * \
                           (abs(data['close'] - data['prev_close']) / data['prev_close']) * \
                           (data['volume'] / data['prev_volume']) * \
                           (abs(data['open'] - data['prev_close']) / data['prev_close']) * \
                           ((data['high'] - data['low']) / (data['prev_high'] - data['prev_low']))
    
    # Volume-Volatility Acceleration (simplified)
    vol_volume_acceleration = (((data['close']/data['prev_close'] - 1) * (data['volume']/data['prev_volume'] - 1)) - \
                              ((data['prev_close']/data['close'].shift(2) - 1) * (data['prev_volume']/data['volume'].shift(2) - 1))) * \
                              ((data['high'] - data['low']) / (data['prev_high'] - data['prev_low'])) * \
                              (data['volume'] / data['prev_volume']) * \
                              (abs(data['open'] - data['prev_close']) / data['prev_close'])
    
    volume_volatility_component = (vol_confirmation_strength + vol_amount_efficiency + vol_volume_acceleration) / 3 * \
                                 (data['amount'] / data['prev_amount']) * \
                                 (abs(data['open'] - data['prev_close']) / data['prev_close'])
    
    # Range-Volatility Component (simplified)
    short_term_range_efficiency = (abs(data['close'] - data['open']) / (data['high'] - data['low'])) * \
                                 (data['volume'] / data['prev_volume']) * \
                                 (abs(data['open'] - data['prev_close']) / data['prev_close']) * \
                                 ((data['high'] - data['low']) / (data['prev_high'] - data['prev_low']))
    
    medium_term_range_asymmetry = ((data['high'] - data['close']) / (data['close'] - data['low'])) * \
                                 (data['amount'] / data['prev_amount']) * \
                                 (abs(data['open'] - data['prev_close']) / (data['high'] - data['low'])) * \
                                 ((data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5)))
    
    # Long-Term Range Compression
    long_term_range_compression = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        window = data.iloc[i-2:i+1]
        avg_range = (window['high'] - window['low']).mean()
        long_term_range_compression.iloc[i] = ((data['high'].iloc[i] - data['low'].iloc[i]) / avg_range) * \
                                             (data['volume'].iloc[i] / data['prev_volume'].iloc[i]) * \
                                             (abs(data['open'].iloc[i] - data['prev_close'].iloc[i]) / (data['prev_high'].iloc[i] - data['prev_low'].iloc[i])) * \
                                             ((data['high'].iloc[i] - data['low'].iloc[i]) / (data['high'].iloc[i-20] - data['low'].iloc[i-20]))
    
    range_volatility_component = (short_term_range_efficiency + medium_term_range_asymmetry + long_term_range_compression) / 3 * \
                                (data['volume'] / data['prev_volume']) * \
                                (abs(data['open'] - data['prev_close']) / data['prev_close'])
    
    # Cross-Scale Alignment (simplified)
    opening_vol_alignment = opening_vol_momentum * medium_term_vol_persistence * np.sign(data['close'] - data['open']) * -1 * \
                           (data['volume'] / data['prev_volume']) * (abs(data['open'] - data['prev_close']) / data['prev_close'])
    
    midday_vol_alignment = midday_vol_efficiency * vol_range_expansion * np.sign(data['close'] - data['open']) * -1 * \
                          (data['amount'] / data['prev_amount']) * (abs(data['open'] - data['prev_close']) / data['prev_close'])
    
    closing_vol_alignment = closing_vol_momentum * vol_price_efficiency * np.sign(data['close'] - data['open']) * -1 * \
                           (data['volume'] / data['prev_volume']) * (abs(data['close'] - data['prev_close']) / data['prev_close'])
    
    short_medium_cross_alignment = (opening_vol_alignment + midday_vol_alignment + closing_vol_alignment) / 3
    
    # Cross-Scale Alignment Multipliers
    short_medium_multiplier = np.where(short_medium_cross_alignment > 0, 1.2, 
                                      np.where(short_medium_cross_alignment < 0, 0.8, 1.0))
    
    # Base Volatility Fractal
    base_vol_fractal = (multi_scale_component + volume_volatility_component + range_volatility_component) / 3 * \
                      (short_medium_multiplier + vol_volume_acceleration + medium_term_range_asymmetry) / 3
    
    # Enhanced Volatility Fractal
    enhanced_vol_fractal = base_vol_fractal * vol_confirmation_strength * short_term_range_efficiency * \
                          medium_term_range_asymmetry * vol_volume_acceleration
    
    # Final Alpha Signal
    final_alpha = enhanced_vol_fractal * vol_price_efficiency * \
                 (abs(data['open'] - data['prev_close']) / data['prev_close']) * \
                 np.sign(data['close'] - data['open'])
    
    # Clean and return
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return final_alpha
