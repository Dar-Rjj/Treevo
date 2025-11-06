import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Regime Persistence
    # Momentum Regime Persistence
    short_term_returns = df['close'].pct_change()
    short_term_sign = np.sign(short_term_returns)
    short_persistence = short_term_sign.rolling(window=3).apply(lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 3 else 0, raw=False)
    
    medium_term_returns = (df['close'] / df['close'].shift(3) - 1)
    medium_term_sign = np.sign(medium_term_returns)
    medium_persistence = medium_term_sign.rolling(window=3).apply(lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 3 else 0, raw=False)
    multi_scale_alignment = short_persistence * medium_persistence
    
    # Volatility Regime Persistence
    daily_ranges = (df['high'] - df['low']) / df['close']
    range_expansion = (daily_ranges > daily_ranges.shift(1)).astype(int)
    volatility_duration = range_expansion.rolling(window=5).sum()
    
    range_expansion_3d = (daily_ranges > daily_ranges.shift(3)).astype(int)
    range_persistence = range_expansion_3d.rolling(window=3).sum()
    volatility_clustering = volatility_duration * range_persistence
    
    # Volume Regime Persistence
    volume_trend = (df['volume'] > df['volume'].shift(1)).astype(int)
    volume_trend_persistence = volume_trend.rolling(window=4).sum()
    
    volume_returns = df['volume'].pct_change()
    volume_sign = np.sign(volume_returns)
    volume_acceleration_consistency = volume_sign.rolling(window=3).apply(lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 3 else 0, raw=False)
    volume_regime_strength = volume_trend_persistence * volume_acceleration_consistency
    
    # Multi-Scale Volatility Structure
    ultra_short_volatility = daily_ranges
    short_volatility = daily_ranges.rolling(window=3).std()
    medium_volatility = daily_ranges.rolling(window=6).std()
    volatility_slope = short_volatility / medium_volatility
    
    # Volatility Regime Classification
    expanding_volatility = (volatility_slope > 1.2).astype(int)
    contracting_volatility = (volatility_slope < 0.8).astype(int)
    stable_volatility = ((volatility_slope >= 0.8) & (volatility_slope <= 1.2)).astype(int)
    
    # Volatility Transition Signals
    volatility_breakout = expanding_volatility * (ultra_short_volatility > short_volatility).astype(int)
    volatility_collapse = contracting_volatility * (ultra_short_volatility < short_volatility).astype(int)
    regime_shift = volatility_breakout - volatility_collapse
    
    # Three-Way Regime Alignment
    # Momentum-Volatility Alignment
    momentum_persistence = multi_scale_alignment
    trending_stable_volatility = momentum_persistence * stable_volatility
    
    momentum_acceleration = short_term_returns.diff()
    acceleration_expanding_vol = momentum_acceleration * expanding_volatility
    
    momentum_reversal = (short_term_sign != short_term_sign.shift(1)).astype(int)
    reversal_high_vol = momentum_reversal * volatility_breakout
    
    # Volume-Momentum Alignment
    confirmed_trend_persistence = momentum_persistence * volume_trend_persistence
    
    volume_divergence = volume_acceleration_consistency * momentum_reversal
    
    volume_supported_acceleration = volume_regime_strength * momentum_acceleration
    
    # Full Three-Way Convergence
    perfect_alignment = multi_scale_alignment * volatility_clustering * volume_regime_strength
    
    momentum_vol_alignment = trending_stable_volatility + acceleration_expanding_vol + reversal_high_vol
    volume_momentum_alignment = confirmed_trend_persistence + volume_divergence + volume_supported_acceleration
    partial_convergence = momentum_vol_alignment * volume_momentum_alignment
    
    regime_conflict = momentum_reversal * volatility_collapse * volume_divergence
    
    # Persistence-Enhanced Factors
    # Regime-Duration Weighted Momentum
    momentum_regime_persistence = multi_scale_alignment.rolling(window=3).mean()
    duration_adjusted_return = medium_term_returns * momentum_regime_persistence
    
    volatility_persisted_acceleration = momentum_acceleration * volatility_duration
    
    volume_confirmed_persistence = duration_adjusted_return * volume_trend_persistence
    
    # Multi-Timeframe Volatility Persistence
    short_vol_persistence = short_volatility * range_persistence
    medium_vol_stability = medium_volatility * volatility_clustering
    volatility_regime_quality = short_vol_persistence / medium_vol_stability.replace(0, np.nan)
    
    # Three-Way Persistence Composite
    momentum_persistence_3d = multi_scale_alignment.rolling(window=3).mean()
    volatility_persistence = volatility_clustering.rolling(window=3).mean()
    volume_persistence = volume_regime_strength.rolling(window=3).mean()
    core_persistence = momentum_persistence_3d * volatility_persistence * volume_persistence
    
    three_way_convergence = perfect_alignment + partial_convergence - regime_conflict
    regime_alignment_persistence = three_way_convergence * core_persistence
    
    persistence_acceleration = regime_alignment_persistence * momentum_acceleration
    
    # Final Alpha Construction
    # Base Persistence Framework
    three_way_persistence_composite = core_persistence + regime_alignment_persistence + persistence_acceleration
    primary_factor = three_way_persistence_composite
    
    adjusted_factor = primary_factor * volatility_regime_quality.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    momentum_enhanced = adjusted_factor * volume_confirmed_persistence
    
    # Regime Transition Overlay
    breakout_signals = momentum_enhanced * volatility_breakout
    collapse_signals = momentum_enhanced * volatility_collapse
    transition_adjusted = breakout_signals - collapse_signals
    
    # Final Alpha Output
    daily_persistence_weighted = momentum_enhanced.rolling(window=5).mean()
    regime_transition_enhanced = transition_adjusted.rolling(window=3).mean()
    three_way_aligned_persistence = perfect_alignment.rolling(window=3).mean()
    
    final_alpha = (daily_persistence_weighted + 
                   regime_transition_enhanced + 
                   three_way_aligned_persistence) / 3
    
    return final_alpha
