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
    vol_confidence = pd.Series(0.7, index=df.index)  # default medium confidence
    vol_confidence[vol_mom_sign_same] = 1.0  # high confidence
    vol_confidence[vol_mom_divergence] = 0.3  # low confidence
    
    # Dynamic Persistence Scoring
    # Direction persistence strength
    mom_5d_direction = np.sign(mom_5d)
    persistence_count = 0
    persistence_scores = []
    
    for i in range(len(df)):
        if i == 0:
            persistence_count = 1
        elif mom_5d_direction.iloc[i] == mom_5d_direction.iloc[i-1]:
            persistence_count += 1
        else:
            persistence_count = 1
        
        # Exponential weighting
        direction_persistence = sum(2**(-j) for j in range(persistence_count))
        persistence_scores.append(direction_persistence)
    
    direction_persistence = pd.Series(persistence_scores, index=df.index)
    
    # Magnitude stability assessment
    mom_2d_std = mom_2d.rolling(window=4, min_periods=1).std()
    magnitude_stability = 1 / (mom_2d_std + 0.0001)
    
    # Combined persistence metric
    persistence_metric = direction_persistence * magnitude_stability * abs(mom_5d)
    
    # Price Range Context Analysis
    # Intraday volatility proxy
    intraday_vol = (df['high'] - df['low']) / df['close']
    
    # Relative range position
    range_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Range-based signal adjustment
    range_adjustment = pd.Series(1.0, index=df.index)
    uptrend_strong = (mom_5d > 0) & (range_position > 0.7)
    downtrend_strong = (mom_5d < 0) & (range_position < 0.3)
    weak_signal = ((mom_5d > 0) & (range_position < 0.3)) | ((mom_5d < 0) & (range_position > 0.7))
    
    range_adjustment[uptrend_strong] = 1.2
    range_adjustment[downtrend_strong] = 1.2
    range_adjustment[weak_signal] = 0.8
    
    # Regime-Adaptive Weighting System
    # Volatility regime detection
    returns = df['close'].pct_change()
    vol_std_10d = returns.rolling(window=10, min_periods=1).std()
    vol_percentile_20d = vol_std_10d.rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    
    # Regime-specific parameter adjustment
    regime_weight = pd.Series(1.0, index=df.index)
    high_vol = vol_percentile_20d > 0.8
    low_vol = vol_percentile_20d < 0.2
    
    # High volatility: reduce momentum weights, increase persistence focus
    regime_weight[high_vol] = 0.8
    
    # Low volatility: emphasize momentum convergence signals
    regime_weight[low_vol] = 1.1
    
    # Final Alpha Generation
    # Core momentum signal
    core_momentum = mom_5d + 0.3 * mom_convergence
    
    # Volume-adapted signal
    volume_adapted = core_momentum * vol_confidence
    
    # Persistence-enhanced signal
    persistence_enhanced = volume_adapted * persistence_metric
    
    # Regime-optimized signal
    regime_optimized = persistence_enhanced * regime_weight
    
    # Range-optimized alpha
    final_alpha = (regime_optimized * range_adjustment) / (intraday_vol + 0.0001)
    
    return final_alpha
