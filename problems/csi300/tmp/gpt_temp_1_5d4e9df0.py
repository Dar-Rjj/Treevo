import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum Persistence with Volume-Price Divergence Detection
    """
    # Multi-Timeframe Momentum Framework
    df = df.copy()
    
    # Short-term momentum (2-day)
    mom_2d = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    
    # Medium-term momentum (5-day)
    mom_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Momentum convergence analysis
    mom_convergence = mom_2d - mom_5d
    
    # Adaptive Volume Confirmation System
    # Volume trend strength
    vol_ma_3 = df['volume'].rolling(window=3, min_periods=1).mean()
    vol_trend = df['volume'] / vol_ma_3
    
    # Volume-price divergence detection
    vol_mom_sign_same = (vol_trend > 1) == (mom_5d > 0)
    vol_mom_divergence = (vol_trend < 1) & (mom_5d > 0)
    vol_mom_divergence = vol_mom_divergence | ((vol_trend > 1) & (mom_5d < 0))
    
    # Volume confidence adjustment
    vol_confidence = np.where(vol_mom_sign_same, 1.0, 
                             np.where(vol_mom_divergence, 0.3, 0.7))
    
    # Dynamic Persistence Scoring
    # Direction persistence strength
    mom_5d_direction = np.sign(mom_5d)
    persistence_count = 0
    persistence_strength = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        if mom_5d_direction.iloc[i] == mom_5d_direction.iloc[i-1] and not pd.isna(mom_5d_direction.iloc[i-1]):
            persistence_count += 1
        else:
            persistence_count = 0
        
        if persistence_count > 0:
            weights = [2**(-j) for j in range(persistence_count)]
            persistence_strength.iloc[i] = sum(weights)
    
    # Magnitude stability assessment
    mom_2d_std = mom_2d.rolling(window=4, min_periods=1).std()
    magnitude_stability = 1 / (mom_2d_std + 0.0001)
    
    # Combined persistence metric
    persistence_metric = persistence_strength * magnitude_stability * np.abs(mom_5d)
    
    # Price Range Context Analysis
    # Intraday volatility proxy
    intraday_vol = (df['high'] - df['low']) / df['close']
    
    # Relative range position
    range_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
    
    # Range-based signal adjustment
    range_adjustment = np.where(
        (mom_5d > 0) & (range_position > 0.7), 1.2,  # strong uptrend, close near high
        np.where((mom_5d < 0) & (range_position < 0.3), 1.2,  # strong downtrend, close near low
                np.where((mom_5d > 0) & (range_position < 0.3), 0.8,  # uptrend but close near low
                        np.where((mom_5d < 0) & (range_position > 0.7), 0.8, 1.0)))  # downtrend but close near high
    )
    
    # Regime-Adaptive Weighting
    # Volatility regime detection
    returns_10d = df['close'].pct_change(periods=10)
    vol_std_10d = returns_10d.rolling(window=10, min_periods=1).std()
    vol_percentile = vol_std_10d.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) if len(x.dropna()) > 0 else False
    )
    
    high_vol_regime = vol_percentile > 0.8
    low_vol_regime = vol_std_10d.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] < np.percentile(x.dropna(), 20)) if len(x.dropna()) > 0 else False
    )
    
    # Momentum regime classification
    trending_regime = np.abs(mom_5d) > 0.02
    mean_reverting_regime = np.abs(mom_5d) < 0.005
    neutral_regime = ~trending_regime & ~mean_reverting_regime
    
    # Adaptive component weights
    persistence_weight = np.where(high_vol_regime, 1.5, 1.0)
    convergence_weight = np.where(low_vol_regime, 1.5, 1.0)
    volume_weight = np.where(trending_regime, 1.5, 1.0)
    overall_strength = np.where(mean_reverting_regime, 0.7, 1.0)
    
    # Final Alpha Generation
    # Core momentum signal
    core_momentum = mom_5d + (mom_convergence * 0.3)
    
    # Volume-adapted signal
    volume_adapted = core_momentum * vol_confidence * volume_weight
    
    # Persistence-enhanced signal
    persistence_enhanced = volume_adapted * (1 + persistence_metric * 0.1) * persistence_weight
    
    # Range-optimized signal
    range_optimized = (persistence_enhanced * range_adjustment) / (intraday_vol + 0.0001)
    
    # Regime-adaptive alpha
    final_alpha = range_optimized * overall_strength * convergence_weight
    
    return final_alpha
