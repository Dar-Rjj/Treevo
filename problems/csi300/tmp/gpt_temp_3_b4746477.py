import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Efficiency Analysis
    # Short-term efficiency: |Close_t - Close_{t-3}| / (max(High_{t-2:t}) - min(Low_{t-2:t}))
    short_term_eff = (abs(data['close'] - data['close'].shift(3))) / (
        data['high'].rolling(window=3, min_periods=3).max() - data['low'].rolling(window=3, min_periods=3).min()
    )
    
    # Medium-term efficiency: |Close_t - Close_{t-8}| / (max(High_{t-7:t}) - min(Low_{t-7:t}))
    medium_term_eff = (abs(data['close'] - data['close'].shift(8))) / (
        data['high'].rolling(window=8, min_periods=8).max() - data['low'].rolling(window=8, min_periods=8).min()
    )
    
    # Long-term efficiency: |Close_t - Close_{t-21}| / (max(High_{t-20:t}) - min(Low_{t-20:t}))
    long_term_eff = (abs(data['close'] - data['close'].shift(21))) / (
        data['high'].rolling(window=21, min_periods=21).max() - data['low'].rolling(window=21, min_periods=21).min()
    )
    
    # Multi-Timeframe Momentum Dynamics
    # Momentum acceleration: (Close_t / Close_{t-3} - 1) - (Close_t / Close_{t-8} - 1)
    mom_accel = (data['close'] / data['close'].shift(3) - 1) - (data['close'] / data['close'].shift(8) - 1)
    
    # Momentum persistence: count(sign(Close_i - Close_{i-1}) consistent from t-5 to t)
    price_diff_sign = np.sign(data['close'].diff())
    mom_persistence = price_diff_sign.rolling(window=6).apply(
        lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 6 else np.nan, raw=False
    )
    
    # Breakout momentum: Close_t / max(High_{t-7:t})
    breakout_mom = data['close'] / data['high'].rolling(window=8, min_periods=8).max()
    
    # Efficiency-Momentum Integration
    # Efficiency-weighted momentum: Momentum acceleration × (Short-term efficiency + Medium-term efficiency)
    eff_weighted_mom = mom_accel * (short_term_eff + medium_term_eff)
    
    # Momentum efficiency ratio: (Close_t - Close_{t-8}) / sum(Volume_i × |Close_i - Close_{i-1}| from t-7 to t)
    price_range = abs(data['close'].diff())
    volume_price_range = (data['volume'] * price_range).rolling(window=8, min_periods=8).sum()
    mom_eff_ratio = (data['close'] - data['close'].shift(8)) / volume_price_range
    
    # Breakout efficiency: Breakout momentum × Long-term efficiency
    breakout_eff = breakout_mom * long_term_eff
    
    # Volume Dynamics Analysis
    # Volume momentum: Volume_t / (sum(Volume_{t-4:t}) / 5)
    vol_mom = data['volume'] / (data['volume'].rolling(window=5, min_periods=5).sum() / 5)
    
    # Volume alignment: sign(Close_t - Close_{t-1}) × sign(Volume_t - Volume_{t-1})
    vol_alignment = np.sign(data['close'].diff()) * np.sign(data['volume'].diff())
    
    # Volume-range intensity: Volume_t / (High_t - Low_t)
    vol_range_intensity = data['volume'] / (data['high'] - data['low'])
    
    # Pressure Dynamics Analysis
    # Buying pressure: (Close_t - Low_t) / (High_t - Low_t)
    buying_pressure = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Multi-timeframe pressure: (Close_t - min(Low_{t-7:t})) / (max(High_{t-7:t}) - min(Low_{t-7:t}))
    multi_timeframe_pressure = (data['close'] - data['low'].rolling(window=8, min_periods=8).min()) / (
        data['high'].rolling(window=8, min_periods=8).max() - data['low'].rolling(window=8, min_periods=8).min()
    )
    
    # Pressure persistence: sum(Buying pressure_{t-2:t}) / 3
    pressure_persistence = buying_pressure.rolling(window=3, min_periods=3).sum() / 3
    
    # Volume-Pressure Integration
    # Volume-pressure alignment: Volume alignment × Pressure persistence
    vol_pressure_alignment = vol_alignment * pressure_persistence
    
    # Pressure-volume efficiency: Buying pressure × Volume-range intensity
    pressure_vol_eff = buying_pressure * vol_range_intensity
    
    # Volume-confirmed pressure: Multi-timeframe pressure × Volume momentum
    vol_confirmed_pressure = multi_timeframe_pressure * vol_mom
    
    # Regime Transition Detection - Efficiency Regime Analysis
    # Efficiency momentum: Short-term efficiency / Medium-term efficiency
    eff_mom = short_term_eff / medium_term_eff
    
    # Efficiency acceleration: (Short-term efficiency - Medium-term efficiency) - (Medium-term efficiency - Long-term efficiency)
    eff_accel = (short_term_eff - medium_term_eff) - (medium_term_eff - long_term_eff)
    
    # Efficiency regime shift: count(efficiency_i > efficiency_{i-1} from t-3 to t)
    eff_increasing = (short_term_eff > short_term_eff.shift(1)).rolling(window=4).sum()
    
    # Momentum Regime Analysis
    # Momentum consistency: correlation between (Close_t/Close_{t-3}-1) and (Close_t/Close_{t-8}-1) over t-5 to t
    mom_3d = (data['close'] / data['close'].shift(3) - 1)
    mom_8d = (data['close'] / data['close'].shift(8) - 1)
    mom_consistency = mom_3d.rolling(window=6).corr(mom_8d)
    
    # Regime strength: abs(Momentum acceleration) × Momentum persistence
    regime_strength = abs(mom_accel) * mom_persistence
    
    # Breakout regime: Breakout momentum × Volume momentum
    breakout_regime = breakout_mom * vol_mom
    
    # Regime Alignment Detection
    # Efficiency-momentum regime: Efficiency momentum × Momentum consistency
    eff_mom_regime = eff_mom * mom_consistency
    
    # Volume-pressure regime: Volume-pressure alignment × Pressure-volume efficiency
    vol_pressure_regime = vol_pressure_alignment * pressure_vol_eff
    
    # Regime transition signal: Efficiency acceleration × Regime strength
    regime_transition = eff_accel * regime_strength
    
    # Dynamic Alpha Synthesis - Core Signal Construction
    # Primary efficiency-momentum: Efficiency-weighted momentum × Momentum efficiency ratio
    primary_eff_mom = eff_weighted_mom * mom_eff_ratio
    
    # Volume-pressure confirmation: Volume-confirmed pressure × Volume-pressure alignment
    vol_pressure_confirmation = vol_confirmed_pressure * vol_pressure_alignment
    
    # Regime enhancement: Breakout efficiency × Efficiency-momentum regime
    regime_enhancement = breakout_eff * eff_mom_regime
    
    # Signal Integration
    # Base alpha: Primary efficiency-momentum × Volume-pressure confirmation
    base_alpha = primary_eff_mom * vol_pressure_confirmation
    
    # Regime adjustment: Base alpha × (1 + Regime transition signal)
    regime_adjustment = base_alpha * (1 + regime_transition)
    
    # Breakout amplification: Regime adjustment × Breakout regime
    breakout_amplification = regime_adjustment * breakout_regime
    
    # Final Alpha Generation
    # Directional enhancement: Breakout amplification × sign(Primary efficiency-momentum)
    directional_enhancement = breakout_amplification * np.sign(primary_eff_mom)
    
    # Volume confirmation: Directional enhancement × abs(Volume-pressure confirmation)
    volume_confirmation = directional_enhancement * abs(vol_pressure_confirmation)
    
    # Final alpha: Volume confirmation × Regime enhancement
    final_alpha = volume_confirmation * regime_enhancement
    
    return final_alpha
