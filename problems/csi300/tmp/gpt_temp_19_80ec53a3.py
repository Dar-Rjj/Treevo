import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-regime momentum persistence with adaptive convergence scoring and volume conviction.
    
    Interpretation:
    - Triple-regime classification (volatility, trend, volume) captures complex market dynamics
    - Momentum persistence measures directional consistency across multiple time horizons
    - Adaptive convergence scoring identifies stocks with aligned momentum signals
    - Volume conviction ensures signals are supported by trading activity intensity
    - Regime-specific momentum weighting optimizes signal relevance to market conditions
    - Positive values indicate strong, persistent bullish momentum across timeframes in favorable regimes
    - Negative values suggest consistent bearish pressure with deteriorating momentum characteristics
    - Economic rationale: Stocks showing regime-appropriate, persistent directional movement 
      with volume confirmation exhibit more predictable short-term return patterns
    """
    
    # Core momentum components across multiple timeframes
    daily_range = df['high'] - df['low']
    
    # Short-term momentum (intraday, overnight, daily)
    intraday_momentum = (df['close'] - df['open']) / (daily_range + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (daily_range + 1e-7)
    daily_momentum = (df['close'] - df['close'].shift(1)) / (daily_range + 1e-7)
    
    # Medium-term momentum (swing, weekly)
    swing_momentum = (df['close'] - df['close'].shift(3)) / (
        df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min() + 1e-7
    )
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (
        df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min() + 1e-7
    )
    
    # Long-term momentum (bi-weekly)
    biweekly_momentum = (df['close'] - df['close'].shift(10)) / (
        df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min() + 1e-7
    )
    
    # Volatility regime classification using adaptive ATR percentiles
    true_range = np.maximum(df['high'] - df['low'],
                           np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    atr_5d = true_range.rolling(window=5).mean()
    atr_20d_median = atr_5d.rolling(window=20).median()
    atr_20d_75pct = atr_5d.rolling(window=20).apply(lambda x: np.percentile(x, 75), raw=True)
    atr_20d_25pct = atr_5d.rolling(window=20).apply(lambda x: np.percentile(x, 25), raw=True)
    
    vol_regime = np.where(atr_5d > atr_20d_75pct, 'high_vol',
                         np.where(atr_5d > atr_20d_median, 'medium_high_vol',
                                np.where(atr_5d > atr_20d_25pct, 'medium_low_vol', 'low_vol')))
    
    # Trend regime classification with multi-timeframe strength assessment
    trend_strength_3d = (df['close'] - df['close'].shift(3)).abs() / df['close'].shift(3)
    trend_strength_8d = (df['close'] - df['close'].shift(8)).abs() / df['close'].shift(8)
    trend_strength_15d = (df['close'] - df['close'].shift(15)).abs() / df['close'].shift(15)
    
    trend_direction_3d = np.sign(df['close'] - df['close'].shift(3))
    trend_direction_8d = np.sign(df['close'] - df['close'].shift(8))
    trend_direction_15d = np.sign(df['close'] - df['close'].shift(15))
    
    # Trend persistence across multiple horizons
    trend_alignment_short = (trend_direction_3d == trend_direction_8d).astype(int)
    trend_alignment_medium = (trend_direction_8d == trend_direction_15d).astype(int)
    trend_alignment_long = (trend_direction_3d == trend_direction_15d).astype(int)
    
    trend_persistence_score = (trend_alignment_short + trend_alignment_medium + trend_alignment_long) / 3.0
    avg_trend_strength = (trend_strength_3d + trend_strength_8d + trend_strength_15d) / 3.0
    
    trend_regime = np.where((avg_trend_strength > 0.05) & (trend_persistence_score > 0.66), 'strong_trend',
                           np.where((avg_trend_strength > 0.025) & (trend_persistence_score > 0.33), 'moderate_trend',
                                  np.where(avg_trend_strength < 0.01, 'no_trend', 'weak_trend')))
    
    # Volume regime with adaptive percentile-based thresholds
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_ratio = df['volume'] / (volume_ma_10 + 1e-7)
    volume_20d_75pct = df['volume'].rolling(window=20).apply(lambda x: np.percentile(x, 75), raw=True)
    volume_20d_25pct = df['volume'].rolling(window=20).apply(lambda x: np.percentile(x, 25), raw=True)
    
    volume_regime = np.where(volume_ratio > 2.0, 'extreme_surge',
                            np.where(volume_ratio > 1.5, 'high_surge',
                                   np.where(volume_ratio > 1.2, 'moderate_surge',
                                          np.where(volume_ratio < 0.6, 'low_vol', 'normal_vol'))))
    
    # Directional persistence across all momentum timeframes
    momentum_signs = pd.DataFrame({
        'intraday': np.sign(intraday_momentum),
        'overnight': np.sign(overnight_momentum),
        'daily': np.sign(daily_momentum),
        'swing': np.sign(swing_momentum),
        'weekly': np.sign(weekly_momentum),
        'biweekly': np.sign(biweekly_momentum)
    })
    
    # Enhanced directional persistence with weighted consistency
    sign_consistency = momentum_signs.apply(lambda x: x.nunique() == 1, axis=1)
    directional_persistence_binary = sign_consistency.astype(int)
    
    # Momentum magnitude persistence (reduced volatility in recent periods)
    momentum_volatility_ratio = (
        (intraday_momentum.rolling(window=3).std() / (intraday_momentum.rolling(window=10).std() + 1e-7)) +
        (daily_momentum.rolling(window=3).std() / (daily_momentum.rolling(window=10).std() + 1e-7)) +
        (swing_momentum.rolling(window=3).std() / (swing_momentum.rolling(window=10).std() + 1e-7))
    ) / 3.0
    
    momentum_magnitude_persistence = (momentum_volatility_ratio < 1.0).astype(int)
    
    # Combined directional persistence score
    combined_directional_persistence = (
        directional_persistence_binary * 0.6 + 
        momentum_magnitude_persistence * 0.4
    )
    
    # Adaptive momentum convergence scoring
    convergence_pairs = [
        (intraday_momentum, swing_momentum),
        (daily_momentum, weekly_momentum),
        (swing_momentum, biweekly_momentum),
        (weekly_momentum, biweekly_momentum),
        (intraday_momentum, biweekly_momentum)
    ]
    
    convergence_scores = []
    for mom1, mom2 in convergence_pairs:
        sign_match = (np.sign(mom1) == np.sign(mom2)).astype(int)
        magnitude_ratio = np.minimum(np.abs(mom1), np.abs(mom2)) / (np.maximum(np.abs(mom1), np.abs(mom2)) + 1e-7)
        convergence_score = sign_match * magnitude_ratio
        convergence_scores.append(convergence_score)
    
    momentum_convergence = pd.concat(convergence_scores, axis=1).mean(axis=1)
    
    # Volume conviction with regime-adaptive scaling
    volume_conviction_base = volume_ratio * np.sign(daily_momentum)
    volume_persistence = (volume_ratio.rolling(window=3).std() < volume_ratio.rolling(window=10).std()).astype(int)
    
    volume_conviction = volume_conviction_base * (1.0 + volume_persistence * 0.3)
    
    # Enhanced regime-adaptive momentum weighting
    regime_weights = {
        'high_vol': {
            'strong_trend': {'intraday': 0.08, 'overnight': 0.06, 'daily': 0.18, 'swing': 0.22, 'weekly': 0.28, 'biweekly': 0.12, 'convergence': 0.06},
            'moderate_trend': {'intraday': 0.12, 'overnight': 0.08, 'daily': 0.20, 'swing': 0.18, 'weekly': 0.24, 'biweekly': 0.10, 'convergence': 0.08},
            'weak_trend': {'intraday': 0.16, 'overnight': 0.12, 'daily': 0.22, 'swing': 0.16, 'weekly': 0.20, 'biweekly': 0.08, 'convergence': 0.06},
            'no_trend': {'intraday': 0.20, 'overnight': 0.15, 'daily': 0.25, 'swing': 0.15, 'weekly': 0.15, 'biweekly': 0.05, 'convergence': 0.05}
        },
        'medium_high_vol': {
            'strong_trend': {'intraday': 0.10, 'overnight': 0.08, 'daily': 0.20, 'swing': 0.24, 'weekly': 0.22, 'biweekly': 0.10, 'convergence': 0.06},
            'moderate_trend': {'intraday': 0.14, 'overnight': 0.10, 'daily': 0.22, 'swing': 0.20, 'weekly': 0.18, 'biweekly': 0.08, 'convergence': 0.08},
            'weak_trend': {'intraday': 0.18, 'overnight': 0.14, 'daily': 0.24, 'swing': 0.16, 'weekly': 0.14, 'biweekly': 0.06, 'convergence': 0.08},
            'no_trend': {'intraday': 0.24, 'overnight': 0.18, 'daily': 0.26, 'swing': 0.12, 'weekly': 0.10, 'biweekly': 0.04, 'convergence': 0.06}
        },
        'medium_low_vol': {
            'strong_trend': {'intraday': 0.12, 'overnight': 0.10, 'daily': 0.22, 'swing': 0.26, 'weekly': 0.18, 'biweekly': 0.08, 'convergence': 0.04},
            'moderate_trend': {'intraday': 0.16, 'overnight': 0.12, 'daily': 0.24, 'swing': 0.22, 'weekly': 0.16, 'biweekly': 0.06, 'convergence': 0.04},
            'weak_trend': {'intraday': 0.20, 'overnight': 0.16, 'daily': 0.26, 'swing': 0.18, 'weekly': 0.12, 'biweekly': 0.04, 'convergence': 0.04},
            'no_trend': {'intraday': 0.28, 'overnight': 0.22, 'daily': 0.28, 'swing': 0.10, 'weekly': 0.08, 'biweekly': 0.02, 'convergence': 0.02}
        },
        'low_vol': {
            'strong_trend': {'intraday': 0.15, 'overnight': 0.12, 'daily': 0.25, 'swing': 0.28, 'weekly': 0.12, 'biweekly': 0.05, 'convergence': 0.03},
            'moderate_trend': {'intraday': 0.20, 'overnight': 0.16, 'daily': 0.26, 'swing': 0.24, 'weekly': 0.10, 'biweekly': 0.03, 'convergence': 0.01},
            'weak_trend': {'intraday': 0.25, 'overnight': 0.20, 'daily': 0.28, 'swing': 0.20, 'weekly': 0.05, 'biweekly': 0.01, 'convergence': 0.01},
            'no_trend': {'intraday': 0.35, 'overnight': 0.25, 'daily': 0.30, 'swing': 0.08, 'weekly': 0.02, 'biweekly': 0.00, 'convergence': 0.00}
        }
    }
    
    volume_multipliers = {
        'extreme_surge': 1.8, 'high_surge': 1.4, 'moderate_surge': 1.2, 'normal_vol': 1.0, 'low_vol': 0.7
    }
    
    # Apply enhanced regime-adaptive weighting
    regime_weighted_momentum = pd.Series(0.0, index=df.index)
    
    for vol_r in ['high_vol', 'medium_high_vol', 'medium_low_vol', 'low_vol']:
        for trend_r in ['strong_trend', 'moderate_trend', 'weak_trend', 'no_trend']:
            regime_mask = (vol_regime == vol_r) & (trend_regime == trend_r)
            if regime_mask.any():
                weights = regime_weights[vol_r][trend_r]
                
                regime_momentum = (
                    weights['intraday'] * intraday_momentum +
                    weights['overnight'] * overnight_momentum +
                    weights['daily'] * daily_momentum +
                    weights['swing'] * swing_momentum +
                    weights['weekly'] * weekly_momentum +
                    weights['biweekly'] * biweekly_momentum +
                    weights['convergence'] * momentum_convergence
                )
                
                regime_weighted_momentum[regime_mask] = regime_momentum[regime_mask]
    
    # Apply volume regime multipliers
    volume_multiplier = pd.Series(1.0, index=df.index)
    for vol_r, multiplier in volume_multipliers.items():
        volume_multiplier[volume_regime == vol_r] = multiplier
    
    # Final alpha factor with enhanced regime optimization
    alpha_factor = (
        regime_weighted_momentum * 
        combined_directional_persistence * 
        volume_multiplier *
        volume_conviction
    )
    
    return alpha_factor
