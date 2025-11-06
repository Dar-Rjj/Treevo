import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate market and sector proxies (using rolling averages as proxies)
    market_close = data['close'].rolling(window=50, min_periods=50).mean()
    sector_close = data['close'].rolling(window=20, min_periods=20).mean()
    
    # Multi-Scale Momentum Fracture Framework
    # Stock vs. Market Momentum Fracture
    stock_market_momentum = ((data['close'] / data['close'].shift(5) - 1) / 
                            (market_close / market_close.shift(5) - 1 + 1e-6) * 
                            (data['high'] - data['low']) / (abs(data['close'] - data['close'].shift(2)) + 1e-6))
    
    # Stock vs. Sector Momentum Fracture
    stock_sector_momentum = ((data['close'] / data['close'].shift(5) - 1) / 
                            (sector_close / sector_close.shift(5) - 1 + 1e-6) * 
                            (data['high'] - data['low']) / (abs(data['open'] - data['close'].shift(2)) + 1e-6))
    
    # Hierarchical Momentum Fracture Divergence
    hierarchical_divergence = stock_market_momentum - stock_sector_momentum
    
    # Volatility-Adjusted Momentum Fracture
    momentum_fracture_vol_adj = hierarchical_divergence * (data['high'] - data['low']) / (abs(data['close'] - data['open']) + 1e-6)
    amount_weighted_momentum = momentum_fracture_vol_adj * data['amount'] / (data['amount'].shift(5) + 1e-6)
    
    # Hierarchical Momentum Fracture Persistence
    def count_sign_consistency(series, window=6):
        current_sign = np.sign(series)
        consistency_count = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            consistency_count.iloc[i] = (np.sign(window_data) == current_sign.iloc[i]).sum()
        return consistency_count
    
    momentum_direction_consistency = count_sign_consistency(hierarchical_divergence, 6)
    momentum_magnitude_persistence = momentum_direction_consistency * abs(hierarchical_divergence) / (data['high'] - data['low'] + 1e-6)
    fracture_momentum_acceleration = (hierarchical_divergence - hierarchical_divergence.shift(3)) / (abs(hierarchical_divergence.shift(3)) + 1e-6)
    
    # Multi-Timeframe Momentum Regime Analysis
    # Volatility Momentum Regime Dynamics
    short_term_momentum_vol = ((data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5) + 1e-6) * 
                              abs(data['close'] - data['close'].shift(3)) / (data['high'] - data['low'] + 1e-6))
    
    medium_term_momentum_vol = ((data['high'] - data['low']) / (data['high'].shift(10) - data['low'].shift(10) + 1e-6) * 
                               abs(data['close'] - data['close'].shift(7)) / 
                               (data['close'].diff().abs().rolling(window=6).sum() + 1e-6))
    
    volatility_momentum_regime_shift = short_term_momentum_vol - medium_term_momentum_vol * np.sign(hierarchical_divergence)
    
    # Volume Momentum Regime Dynamics
    momentum_volume_velocity = (data['volume'] / (data['volume'].shift(3) + 1e-6) - 1) * np.sign(data['close'] - data['close'].shift(2))
    momentum_volume_intensity = (np.log(data['volume'] + 1e-6) / 
                               np.log(data['volume'].rolling(window=7).mean() + 1e-6) * 
                               abs((data['close'] - data['close'].shift(2)) / (data['close'].shift(2) + 1e-6)))
    volume_momentum_divergence = momentum_volume_velocity - momentum_volume_intensity * np.sign(data['close'] - data['close'].shift(1))
    
    # Price Efficiency Momentum Dynamics
    opening_momentum_efficiency = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6) * 
                                 (data['open'] - data['close'].shift(2)) / (abs(data['open'] - data['close'].shift(3)) + 1e-6))
    
    intraday_momentum_efficiency = ((data['high'] - data['close']) / (data['close'] - data['low'] + 1e-6) * 
                                   (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6))
    
    momentum_efficiency_ratio = opening_momentum_efficiency / (intraday_momentum_efficiency + 1e-6)
    
    # Momentum Structure with Fracture Integration
    # Opening Momentum Fracture
    hierarchical_opening_pressure = ((data['open'] - data['low']) / (data['high'] - data['open'] + 1e-6) - 1) * np.sign(hierarchical_divergence)
    opening_momentum_strength = ((data['open'] - data['close'].shift(2)) * (data['close'] - data['open']) * 
                               abs(data['close'].shift(2) - data['open']) / (data['high'] - data['low'] + 1e-6))
    opening_fracture_momentum = opening_momentum_strength * hierarchical_opening_pressure
    
    # Intraday Momentum Fracture
    hierarchical_intraday_micro = hierarchical_divergence * (data['close'] - data['close'].shift(2)) / (data['high'] - data['low'] + 1e-6)
    intraday_momentum_intensity = (np.log(data['high'] - data['low'] + 1e-6) / 
                                 np.log(abs(data['close'] - (data['high'] + data['low']) / 2) + 1e-6) * 
                                 abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'] + 1e-6))
    intraday_fracture_momentum = intraday_momentum_intensity * hierarchical_intraday_micro
    
    # Closing Momentum Fracture
    hierarchical_closing_momentum = hierarchical_divergence * data['amount'] / (data['amount'].shift(2) + 1e-6)
    closing_momentum_strength = (abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6) * 
                               ((data['close'] - data['close'].shift(2)) - (data['close'].shift(4) - data['close'].shift(6))))
    closing_fracture_momentum = closing_momentum_strength * hierarchical_closing_momentum
    
    # Dynamic Momentum Regime-Shift Detection
    # Volatility Momentum Regime Classification
    high_vol_regime = (short_term_momentum_vol > 1.5) & (medium_term_momentum_vol > 1.3)
    low_vol_regime = (short_term_momentum_vol < 0.6) & (medium_term_momentum_vol < 0.7)
    normal_vol_regime = ~high_vol_regime & ~low_vol_regime
    
    # Volume Momentum Regime Classification
    high_volume_regime = (momentum_volume_velocity > 1.7) & (momentum_volume_intensity > 1.4)
    low_volume_regime = (momentum_volume_velocity < 0.5) & (momentum_volume_intensity < 0.6)
    normal_volume_regime = ~high_volume_regime & ~low_volume_regime
    
    # Efficiency Momentum Regime Classification
    high_efficiency_regime = (opening_momentum_efficiency > 0.9) & (intraday_momentum_efficiency > 1.3)
    low_efficiency_regime = (opening_momentum_efficiency < 0.15) & (intraday_momentum_efficiency < 0.7)
    normal_efficiency_regime = ~high_efficiency_regime & ~low_efficiency_regime
    
    # Regime Amplifiers
    vol_amplifier = pd.Series(1.0, index=data.index)
    vol_amplifier[high_vol_regime] = 0.45
    vol_amplifier[low_vol_regime] = 0.12
    
    volume_amplifier = pd.Series(1.0, index=data.index)
    volume_amplifier[high_volume_regime] = 0.3
    volume_amplifier[low_volume_regime] = 0.07
    
    efficiency_amplifier = pd.Series(1.0, index=data.index)
    efficiency_amplifier[high_efficiency_regime] = 0.35
    efficiency_amplifier[low_efficiency_regime] = 0.1
    
    combined_regime_amplifier = vol_amplifier * volume_amplifier * efficiency_amplifier
    
    # Final Hierarchical Momentum Fracture Alpha
    core_momentum_fracture = (opening_fracture_momentum * intraday_fracture_momentum * 
                             closing_fracture_momentum * hierarchical_divergence)
    
    amount_enhancement = (core_momentum_fracture * data['amount'] / (data['amount'].shift(5) + 1e-6) * 
                         momentum_magnitude_persistence)
    
    fracture_momentum_acceleration_final = amount_enhancement * fracture_momentum_acceleration
    
    final_alpha = fracture_momentum_acceleration_final * combined_regime_amplifier
    
    return final_alpha
