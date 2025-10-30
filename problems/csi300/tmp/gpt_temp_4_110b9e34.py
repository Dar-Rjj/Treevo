import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Scale Efficiency Analysis
    df['micro_efficiency'] = (df['close'] - df['close'].shift(2)) / (df['high'].rolling(2).max() - df['low'].rolling(2).min())
    df['meso_efficiency'] = (df['close'] - df['close'].shift(8)) / (df['high'].rolling(8).max() - df['low'].rolling(8).min())
    df['macro_efficiency'] = (df['close'] - df['close'].shift(21)) / (df['high'].rolling(21).max() - df['low'].rolling(21).min())
    df['fractal_efficiency_coherence'] = df['micro_efficiency'] * df['meso_efficiency'] * df['macro_efficiency']
    
    # Intraday Range Assessment
    df['daily_range_efficiency'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['opening_gap_efficiency'] = (df['open'] - df['low']) / (df['high'] - df['low'])
    session_strength_asymmetry = np.abs(df['daily_range_efficiency'] - 0.5) - np.abs(df['opening_gap_efficiency'] - 0.5)
    df['range_utilization_consistency'] = df['daily_range_efficiency'] * session_strength_asymmetry
    
    # Volume-Liquidity Dynamics
    df['volume_acceleration'] = df['volume'] / df['volume'].shift(3)
    df['dollar_flow_intensity'] = df['amount'] / df['amount'].shift(3)
    df['regime_intensity'] = df['volume_acceleration'] * df['dollar_flow_intensity']
    
    # Momentum Quality Assessment
    short_momentum = np.sign(df['close'] - df['close'].shift(3))
    medium_momentum = np.sign(df['close'] - df['close'].shift(8))
    df['momentum_alignment'] = short_momentum * medium_momentum
    df['momentum_acceleration'] = (df['close'] - df['close'].shift(3)) / (df['close'].shift(3) - df['close'].shift(8))
    
    volume_trend_persistence = df['volume'].rolling(5).apply(lambda x: np.corrcoef(range(5), x)[0,1] if len(x) == 5 else np.nan)
    five_day_return = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['volume_weighted_momentum'] = five_day_return * volume_trend_persistence
    
    # Regime-Adaptive Signal Construction
    intensity_threshold = df['regime_intensity'].rolling(10).median()
    high_intensity_regime = (df['regime_intensity'] > intensity_threshold)
    low_intensity_regime = (df['regime_intensity'] <= intensity_threshold)
    
    range_expansion_momentum = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    volume_momentum_accumulation = df['volume_weighted_momentum'].rolling(3).sum()
    
    regime_signal = pd.Series(index=df.index, dtype=float)
    regime_signal[high_intensity_regime] = df['fractal_efficiency_coherence'] * df['regime_intensity']
    regime_signal[low_intensity_regime] = df['fractal_efficiency_coherence'] * df['range_utilization_consistency']
    regime_signal[~high_intensity_regime & ~low_intensity_regime] = range_expansion_momentum * volume_momentum_accumulation
    
    # Multi-Scale Validation
    df['efficiency_consistency'] = df['micro_efficiency'] * df['meso_efficiency'] * df['macro_efficiency']
    df['volume_momentum_alignment'] = df['volume_weighted_momentum'] * df['momentum_acceleration']
    df['range_quality_validation'] = range_expansion_momentum * df['daily_range_efficiency']
    
    # Dynamic Signal Integration
    core_efficiency_signal = df['fractal_efficiency_coherence'] * df['range_quality_validation']
    volume_confirmation = df['volume_momentum_alignment'] * df['regime_intensity']
    adaptive_weighting = core_efficiency_signal * volume_confirmation
    
    # Volatility Adjustment
    signal_persistence = core_efficiency_signal.rolling(3).apply(lambda x: (x > 0).sum())
    efficiency_volatility = df['fractal_efficiency_coherence'].rolling(5).std()
    volatility_calibration = adaptive_weighting / efficiency_volatility
    
    # Final Alpha Synthesis
    liquidity_momentum = df['dollar_flow_intensity'].rolling(3).apply(lambda x: np.polyfit(range(3), x, 1)[0] if len(x) == 3 else np.nan)
    volume_8d_avg = df['volume'].rolling(8).mean()
    volume_scaling = volatility_calibration * (df['volume'] / volume_8d_avg)
    range_positioning = 1 - np.abs(df['daily_range_efficiency'] - 0.5)
    
    # Final alpha factor
    alpha = liquidity_momentum * volume_scaling * range_positioning * signal_persistence
    
    return alpha.fillna(0)
