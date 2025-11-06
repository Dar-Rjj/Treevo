import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Small epsilon to avoid division by zero
    eps = 1e-8
    
    # Calculate basic volatility measures
    micro_vol = df['high'] - df['low']
    
    # Calculate rolling windows for meso and macro volatility
    high_roll_2 = df['high'].rolling(window=3, min_periods=3).max()
    low_roll_2 = df['low'].rolling(window=3, min_periods=3).min()
    meso_vol = high_roll_2 - low_roll_2
    
    high_roll_5 = df['high'].rolling(window=6, min_periods=6).max()
    low_roll_5 = df['low'].rolling(window=6, min_periods=6).min()
    macro_vol = high_roll_5 - low_roll_5
    
    # Volatility Fractal Ratio
    vol_fractal_ratio = (micro_vol * meso_vol) / (macro_vol + eps)
    
    # Volatility Persistence
    micro_vol_avg_5 = micro_vol.rolling(window=5, min_periods=5).mean()
    vol_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(6, len(df)):
        if i >= 6:
            window_data = micro_vol.iloc[i-3:i]
            avg_data = micro_vol_avg_5.iloc[i-3:i]
            count = ((window_data > avg_data).sum()) / 3.0
            vol_persistence.iloc[i] = count
    
    # Volatility Expansion
    vol_expansion = micro_vol / (micro_vol_avg_5 + eps)
    
    # Fractal Volatility Score
    fractal_vol_score = vol_fractal_ratio * vol_persistence * vol_expansion
    
    # Multi-Scale Asymmetric Efficiency
    micro_asymmetry = ((df['high'] - df['open']) - (df['open'] - df['low'])) / (micro_vol + eps)
    meso_asymmetry = ((high_roll_2 - df['open']) - (df['open'] - low_roll_2)) / (meso_vol + eps)
    macro_asymmetry = ((high_roll_5 - df['open']) - (df['open'] - low_roll_5)) / (macro_vol + eps)
    fractal_asymmetric_cascade = micro_asymmetry * meso_asymmetry * macro_asymmetry
    
    # Volume-Fractal Asymmetry
    volume_dist_skew = ((df['high'] - df['open']) * df['volume'] - (df['open'] - df['low']) * df['volume']) / (micro_vol + eps)
    
    # Volume Persistence Ratio
    vol_avg_5 = df['volume'].rolling(window=5, min_periods=5).mean()
    vol_persistence_ratio = pd.Series(index=df.index, dtype=float)
    for i in range(6, len(df)):
        if i >= 6:
            window_data = df['volume'].iloc[i-3:i]
            avg_data = vol_avg_5.iloc[i-3:i]
            count = ((window_data > avg_data).sum()) / 3.0
            vol_persistence_ratio.iloc[i] = count
    
    fractal_volume_asymmetry = volume_dist_skew * vol_persistence_ratio
    
    # Multi-Scale Efficiency Patterns
    micro_efficiency = (df['close'] - df['open']) / (micro_vol + eps)
    meso_efficiency = (df['close'] - df['open']) / (meso_vol + eps)
    macro_efficiency = (df['close'] - df['open']) / (macro_vol + eps)
    efficiency_transmission = (micro_efficiency - meso_efficiency) * (meso_efficiency - macro_efficiency)
    
    # Asymmetry-Efficiency Transmission
    short_term_transmission = pd.Series(index=df.index, dtype=float)
    medium_term_transmission = pd.Series(index=df.index, dtype=float)
    
    for i in range(6, len(df)):
        if i >= 6:
            # Short-term transmission
            micro_asym_window = micro_asymmetry.iloc[i-3:i]
            micro_eff_window = micro_efficiency.iloc[i-3:i]
            short_count = ((np.sign(micro_asym_window) == np.sign(micro_eff_window)).sum()) / 3.0
            
            # Medium-term transmission
            meso_asym_window = meso_asymmetry.iloc[i-3:i]
            meso_eff_window = meso_efficiency.iloc[i-3:i]
            medium_count = ((np.sign(meso_asym_window) == np.sign(meso_eff_window)).sum()) / 3.0
            
            short_term_transmission.iloc[i] = short_count
            medium_term_transmission.iloc[i] = medium_count
    
    fractal_transmission_convergence = short_term_transmission * medium_term_transmission
    
    # Amount-Driven Fractal Microstructure
    avg_trade_size = df['amount'] / (df['volume'] + eps)
    prev_avg_trade_size = avg_trade_size.shift(1)
    trade_size_fractal = (avg_trade_size - prev_avg_trade_size) / (prev_avg_trade_size + eps)
    trade_size_fractal_pressure = trade_size_fractal * (df['close'] - df['open']) / (micro_vol + eps)
    
    # Amount-Volume Fractal Integration
    amount_per_unit_vol = df['amount'] / (df['volume'] + eps)
    vol_per_unit_amount = df['volume'] / (df['amount'] + eps)
    amount_volume_fractal = amount_per_unit_vol * vol_per_unit_amount * np.sign(volume_dist_skew)
    
    # Fractal Transmission Quality Assessment
    volatility_fractal_transmission = vol_fractal_ratio * fractal_volume_asymmetry
    efficiency_fractal_transmission = efficiency_transmission * fractal_transmission_convergence
    fractal_transmission_cascade = volatility_fractal_transmission * efficiency_fractal_transmission
    
    # Fractal Regime Alignment
    fractal_regime_alignment = (np.sign(fractal_asymmetric_cascade) * 
                               np.sign(efficiency_transmission) * 
                               np.sign(amount_volume_fractal))
    
    # Fractal Transmission Consistency
    fractal_transmission_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(1, len(df)):
        if i == 1:
            fractal_transmission_consistency.iloc[i] = 1
        else:
            current_sign = np.sign(fractal_transmission_cascade.iloc[i])
            prev_sign = np.sign(fractal_transmission_cascade.iloc[i-1])
            if current_sign == prev_sign and not pd.isna(fractal_transmission_consistency.iloc[i-1]):
                fractal_transmission_consistency.iloc[i] = fractal_transmission_consistency.iloc[i-1] + 1
            else:
                fractal_transmission_consistency.iloc[i] = 1
    
    fractal_transmission_quality = (fractal_regime_alignment * 
                                   fractal_transmission_consistency * 
                                   vol_persistence_ratio)
    
    # Volatility-Adaptive Signal Generation
    volatility_expansion_signal = (vol_expansion > 1).astype(float) * volume_dist_skew
    fractal_momentum_signal = fractal_asymmetric_cascade * vol_persistence_ratio
    efficiency_transmission_signal = efficiency_transmission * amount_volume_fractal
    
    volatility_compression_signal = (vol_expansion < 1).astype(float) * vol_persistence
    transmission_confirmation = fractal_transmission_convergence * vol_persistence_ratio
    efficiency_persistence = micro_efficiency * meso_efficiency * macro_efficiency
    
    # Composite Alpha Construction
    volatility_fractal_core = fractal_asymmetric_cascade * fractal_vol_score
    efficiency_transmission_core = efficiency_transmission * fractal_transmission_convergence
    volume_amount_core = fractal_volume_asymmetry * amount_volume_fractal
    
    # Regime-Weighted Enhancement
    high_vol_component = ((volatility_expansion_signal + 
                          fractal_momentum_signal + 
                          efficiency_transmission_signal) * vol_expansion)
    
    low_vol_component = ((volatility_compression_signal + 
                         transmission_confirmation + 
                         efficiency_persistence) * vol_persistence)
    
    regime_adaptive_signal = high_vol_component + low_vol_component
    
    # Final Alpha Synthesis
    base_alpha = volatility_fractal_core * efficiency_transmission_core * volume_amount_core
    enhanced_alpha = (base_alpha * regime_adaptive_signal * 
                     fractal_transmission_quality * trade_size_fractal_pressure)
    
    return enhanced_alpha
