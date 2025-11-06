import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Add small epsilon to avoid division by zero
    eps = 0.001
    
    # Calculate basic components
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    volume = df['volume']
    
    # Multi-Scale Fractal Dynamics
    # Micro Range Fractal
    micro_fractal = ((close - close.shift(1)) / (high - low + eps)) * \
                   ((close - close.shift(2)) / (high.shift(2) - low.shift(2) + eps))
    
    # Meso Range Fractal
    high_5d = high.rolling(window=6, min_periods=1).max()
    low_5d = low.rolling(window=6, min_periods=1).min()
    high_8d = high.rolling(window=9, min_periods=1).max()
    low_8d = low.rolling(window=9, min_periods=1).min()
    
    meso_fractal = ((close - close.shift(5)) / (high_5d - low_5d + eps)) * \
                   ((close - close.shift(8)) / (high_8d - low_8d + eps))
    
    # Macro Range Fractal
    high_13d = high.rolling(window=14, min_periods=1).max()
    low_13d = low.rolling(window=14, min_periods=1).min()
    high_21d = high.rolling(window=22, min_periods=1).max()
    low_21d = low.rolling(window=22, min_periods=1).min()
    
    macro_fractal = ((close - close.shift(13)) / (high_13d - low_13d + eps)) * \
                    ((close - close.shift(21)) / (high_21d - low_21d + eps))
    
    # Volatility-Weighted Microstructure
    opening_vol_sig = (high - low) / (open_price + eps)
    closing_vol_sig = (high - low) / (close + eps)
    
    vol_efficiency = ((close - open_price) / (high - low + eps)) * \
                    ((high - low) / (high_5d - low_5d + eps))
    
    # Volume Flow Fractal Synchronization
    morning_vol_intensity = volume * (open_price - low) / (high - low + eps)
    afternoon_vol_intensity = volume * (high - close) / (high - low + eps)
    vol_dist_asymmetry = morning_vol_intensity - afternoon_vol_intensity
    
    vol_range_efficiency = (volume / (high - low + eps)) / \
                          (volume.shift(1) / (high.shift(1) - low.shift(1) + eps))
    
    # Fractal Momentum Structure
    opening_pressure = (close - open_price) * volume / (high - low + eps)
    
    # Adaptive Opening Momentum
    opening_pressure_pos_count = opening_pressure.rolling(window=4, min_periods=1).apply(
        lambda x: np.sum(x > 0), raw=True
    )
    adaptive_opening_momentum = opening_pressure * opening_pressure_pos_count
    
    # Volume Flow Momentum
    vol_asymmetry_sign_count = vol_dist_asymmetry.rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) if len(x) > 0 else 0, raw=False
    )
    vol_flow_momentum = vol_dist_asymmetry * vol_asymmetry_sign_count
    
    # Fractal Momentum Cascade
    fractal_momentum_cascade = micro_fractal * meso_fractal * macro_fractal * np.sign(adaptive_opening_momentum)
    
    # Range Persistence Microstructure
    # Range Momentum Persistence
    range_diff = (high - low) - (high.shift(1) - low.shift(1))
    range_momentum_persistence = range_diff.rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x > 0) - np.sum(x < 0), raw=True
    )
    
    # Volume Range Persistence
    vol_range_current = volume / (high - low + eps)
    vol_range_prev = volume.shift(1) / (high.shift(1) - low.shift(1) + eps)
    vol_range_diff = vol_range_current - vol_range_prev
    vol_range_persistence = vol_range_diff.rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x > 0) - np.sum(x < 0), raw=True
    )
    
    # Volatility Persistence Score
    high_2d = high.rolling(window=3, min_periods=1).max()
    low_2d = low.rolling(window=3, min_periods=1).min()
    high_5d_prev = high.shift(3).rolling(window=3, min_periods=1).max()
    low_5d_prev = low.shift(3).rolling(window=3, min_periods=1).min()
    
    vol_persistence_count = ((high_2d - low_2d) > (high_5d_prev - low_5d_prev)).rolling(
        window=6, min_periods=1
    ).sum()
    
    volatility_persistence_score = vol_persistence_count * \
                                 ((high_2d - low_2d) / (high_5d_prev - low_5d_prev + eps))
    
    # Microstructure Quality Framework
    # Pattern Consistency
    fractal_consistency = (np.sign(micro_fractal) == np.sign(meso_fractal)).rolling(
        window=6, min_periods=1
    ).sum()
    
    momentum_consistency = (np.sign(adaptive_opening_momentum) == np.sign(vol_flow_momentum)).rolling(
        window=6, min_periods=1
    ).sum()
    
    # Volume Quality
    vol_flow_consistency = (np.sign(vol_flow_momentum) == np.sign(vol_flow_momentum.shift(1))).rolling(
        window=6, min_periods=1
    ).sum()
    
    vol_sign_changes = (np.sign(vol_flow_momentum) != np.sign(vol_flow_momentum.shift(1))).rolling(
        window=6, min_periods=1
    ).sum()
    vol_pattern_stability = vol_flow_consistency / (vol_sign_changes + 1)
    
    # Multi-Timeframe Integration
    # Short-term Synchronization
    fractal_vol_alignment = np.sign(micro_fractal) * np.sign(vol_range_efficiency)
    momentum_regime_congruence = np.sign(adaptive_opening_momentum) * \
                               np.sign(opening_vol_sig - closing_vol_sig)
    
    # Medium-term Persistence
    vol_asymmetry_sign_consistency = (np.sign(vol_dist_asymmetry) == np.sign(vol_dist_asymmetry.shift(1))).rolling(
        window=5, min_periods=1
    ).sum()
    multi_day_range_persistence = volatility_persistence_score * vol_asymmetry_sign_consistency
    
    fractal_momentum_alignment = np.sign(fractal_momentum_cascade) * np.sign(range_momentum_persistence)
    
    # Final Alpha Construction
    # Core Signal Components
    fractal_momentum_component = fractal_momentum_cascade * (fractal_consistency * momentum_consistency)
    vol_sync_component = vol_flow_momentum * vol_range_efficiency * (vol_flow_consistency * vol_pattern_stability)
    range_quality_component = (fractal_consistency * momentum_consistency) * (vol_flow_consistency * vol_pattern_stability)
    
    # Signal Integration
    fractal_vol_synthesis = fractal_momentum_component * vol_sync_component
    quality_range_enhancement = fractal_vol_synthesis * range_quality_component
    
    # Multi-Timeframe Integration
    multi_timeframe_integration = (fractal_vol_alignment * momentum_regime_congruence) * \
                                 (multi_day_range_persistence * fractal_momentum_alignment)
    
    # Final Alpha
    alpha = quality_range_enhancement * multi_timeframe_integration
    
    # Fill result series
    result = alpha.fillna(0)
    
    return result
