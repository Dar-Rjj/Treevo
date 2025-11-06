import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Asymmetric Volatility Dynamics
    # Avoid division by zero
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, np.nan)
    
    # Upward Volatility Efficiency
    upward_vol_eff = ((data['high'] - data['open']) / high_low_range) * (data['volume'] / data['amount'])
    
    # Downward Volatility Efficiency
    downward_vol_eff = ((data['open'] - data['low']) / high_low_range) * (data['volume'] / data['amount'])
    
    # Volatility Asymmetry Ratio
    vol_asymmetry_ratio = (upward_vol_eff - downward_vol_eff) * np.sign(data['close'] - data['open'])
    
    # Multi-Timeframe Volume Pressure
    # Immediate Volume Pressure
    volume_ratio = data['volume'] / data['volume'].shift(1)
    price_efficiency = (data['close'] - data['open']) / high_low_range
    immediate_volume_pressure = volume_ratio * price_efficiency
    
    # Short-Term Volume Accumulation (5-day window)
    def calc_volume_accumulation(window):
        volume_signed = window['volume'] * np.sign(window['close'] - window['open'])
        return volume_signed.sum() / window['volume'].sum() if window['volume'].sum() > 0 else 0
    
    short_term_volume_accum = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            short_term_volume_accum.iloc[i] = calc_volume_accumulation(window)
    
    # Medium-Term Volume Trend (20-day window)
    medium_term_volume_trend = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 19:
            window = data.iloc[i-19:i+1]
            medium_term_volume_trend.iloc[i] = calc_volume_accumulation(window)
    
    # Price-Volume Fractal Divergence
    # Intraday Price Efficiency
    intraday_price_efficiency = (data['close'] - data['open']) / high_low_range
    
    # Volume-Weighted Price Momentum (5-day window)
    volume_weighted_momentum = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            weighted_sum = ((window['close'] - window['open']) * window['volume']).sum()
            volume_sum = window['volume'].sum()
            volume_weighted_momentum.iloc[i] = weighted_sum / volume_sum if volume_sum > 0 else 0
    
    # Fractal Divergence Signal
    fractal_divergence_signal = intraday_price_efficiency - volume_weighted_momentum
    
    # Regime-Specific Alpha Components
    high_vol_regime = (vol_asymmetry_ratio > 0.3) & (immediate_volume_pressure > 0.2)
    low_vol_accumulation = (vol_asymmetry_ratio < -0.2) & (medium_term_volume_trend > 0.1)
    transition_phase = (vol_asymmetry_ratio.abs() < 0.1) & (fractal_divergence_signal > 0.15)
    
    # Hierarchical Alpha Construction
    # Volatility Component
    volatility_component = (upward_vol_eff * 0.4 + 
                           downward_vol_eff * 0.3 + 
                           vol_asymmetry_ratio * 0.3)
    
    # Volume Component
    volume_component = (immediate_volume_pressure * 0.4 + 
                       short_term_volume_accum * 0.3 + 
                       medium_term_volume_trend * 0.3)
    
    # Divergence Component
    divergence_component = (intraday_price_efficiency * 0.3 + 
                           volume_weighted_momentum * 0.4 + 
                           fractal_divergence_signal * 0.3)
    
    # Regime Multiplier
    regime_multiplier = pd.Series(1.0, index=data.index)
    regime_multiplier[high_vol_regime] = 1.5
    regime_multiplier[low_vol_accumulation] = 1.2
    regime_multiplier[transition_phase] = 0.9
    
    # Final Alpha
    final_alpha = (volatility_component * volume_component * divergence_component) * regime_multiplier
    
    # Handle NaN values that may occur due to division by zero or insufficient data
    final_alpha = final_alpha.fillna(0)
    
    return final_alpha
