import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Momentum with Microstructure Stress factor
    """
    # Price-Based Volatility Compression
    daily_vol_range = (df['high'] - df['low']) / df['close']
    vol_5d_ma = daily_vol_range.rolling(window=5, min_periods=3).mean()
    vol_compression_ratio = daily_vol_range / vol_5d_ma
    
    # Volume-Volatility Divergence
    volume_change = df['volume'].pct_change()
    vol_change = daily_vol_range.pct_change()
    vol_vol_divergence = volume_change - vol_change
    abnormal_vol_vol = vol_vol_divergence.rolling(window=5, min_periods=3).std()
    
    # Multi-Scale Momentum Persistence
    mom_3d = df['close'].pct_change(periods=3)
    mom_8d = df['close'].pct_change(periods=8)
    momentum_alignment = np.sign(mom_3d) * np.sign(mom_8d)
    momentum_regime_transition = mom_3d.rolling(window=5, min_periods=3).std()
    
    # Momentum-Volume Coordination
    volume_trend = df['volume'].rolling(window=5, min_periods=3).mean()
    volume_momentum_coordination = np.sign(mom_3d) * np.sign(df['volume'].pct_change())
    momentum_confirmation = (mom_3d * volume_momentum_coordination).rolling(window=5, min_periods=3).mean()
    
    # Dynamic Amount Pressure
    amount_flow_intensity = df['amount'] / df['amount'].rolling(window=5, min_periods=3).mean()
    amount_concentration = df['amount'].rolling(window=3, min_periods=2).std() / df['amount'].rolling(window=3, min_periods=2).mean()
    
    # Amount Flow Regime Boundaries
    amount_flow_compression = amount_flow_intensity.rolling(window=5, min_periods=3).std()
    amount_stress_accumulation = amount_flow_intensity.rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(np.where(x < x.mean(), 1, 0)) / len(x)
    )
    
    # Multi-Dimensional Integration
    volatility_stress = vol_compression_ratio * abnormal_vol_vol
    momentum_stress = momentum_alignment * momentum_regime_transition
    volatility_momentum_combination = volatility_stress * momentum_stress
    
    amount_confirmation = amount_flow_intensity * momentum_confirmation
    regime_adaptive_signal = volatility_momentum_combination * amount_confirmation
    
    # Final factor with regime adaptation
    factor = (regime_adaptive_signal * 
              (1 + amount_stress_accumulation) * 
              (1 + amount_flow_compression))
    
    return factor
