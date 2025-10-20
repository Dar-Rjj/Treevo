import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Momentum Efficiency with Volume-Volatility Regime Switching factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate required technical indicators
    df['returns'] = df['close'].pct_change()
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Multi-Scale Momentum Efficiency Analysis
    # 5-day momentum
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_5d_sign'] = np.sign(df['momentum_5d'])
    
    # 15-day momentum
    df['momentum_15d'] = df['close'] / df['close'].shift(15) - 1
    df['momentum_15d_sign'] = np.sign(df['momentum_15d'])
    
    # Momentum sign consistency (5-day)
    df['momentum_5d_consistency'] = (
        df['momentum_5d_sign'].rolling(window=5, min_periods=3)
        .apply(lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0)
    )
    
    # Momentum sign consistency (15-day)
    df['momentum_15d_consistency'] = (
        df['momentum_15d_sign'].rolling(window=15, min_periods=8)
        .apply(lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0)
    )
    
    # Momentum persistence ratio
    df['momentum_persistence_ratio'] = (
        df['momentum_5d_consistency'] / df['momentum_15d_consistency']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Momentum efficiency metrics
    df['price_range_5d'] = (
        df['high'].rolling(window=5).max() / df['low'].rolling(window=5).min() - 1
    )
    df['momentum_efficiency'] = (
        abs(df['momentum_5d']) / df['price_range_5d']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    df['volatility_5d'] = df['true_range'].rolling(window=5).mean()
    df['momentum_per_volatility'] = (
        abs(df['momentum_5d']) / df['volatility_5d']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume-Volatility Regime Classification
    # Volume analysis
    df['volume_percentile'] = (
        df['volume'].rolling(window=10, min_periods=5)
        .apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    )
    df['volume_5d_avg'] = df['volume'].rolling(window=5).mean()
    df['volume_acceleration'] = df['volume'] / df['volume_5d_avg']
    
    # Volume regime classification
    df['volume_regime'] = 0  # normal
    df.loc[df['volume_percentile'] < 0.3, 'volume_regime'] = -1  # low
    df.loc[df['volume_percentile'] > 0.7, 'volume_regime'] = 1   # high
    
    # Volatility regime
    df['volatility_10d_avg'] = df['true_range'].rolling(window=10).mean()
    df['volatility_percentile'] = (
        df['true_range'].rolling(window=10, min_periods=5)
        .apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    )
    
    df['volatility_regime'] = 0  # normal
    df.loc[df['volatility_percentile'] < 0.3, 'volatility_regime'] = -1  # low
    df.loc[df['volatility_percentile'] > 0.7, 'volatility_regime'] = 1   # high
    
    # Regime persistence
    df['volume_regime_persistence'] = (
        df['volume_regime'].rolling(window=5, min_periods=3)
        .apply(lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0)
    )
    df['volatility_regime_persistence'] = (
        df['volatility_regime'].rolling(window=5, min_periods=3)
        .apply(lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0)
    )
    
    # Regime-Adaptive Signal Generation
    # Base momentum signal
    df['base_momentum_signal'] = df['momentum_5d'] * df['momentum_efficiency']
    
    # Regime-specific adjustments
    # High volatility regime: reduce signal sensitivity
    volatility_adjustment = np.where(
        df['volatility_regime'] == 1, 0.7,
        np.where(df['volatility_regime'] == -1, 1.3, 1.0)
    )
    
    # Volume regime confirmation
    volume_confirmation = np.where(
        df['volume_regime'] == np.sign(df['momentum_5d']), 1.2,
        np.where(df['volume_regime'] == -np.sign(df['momentum_5d']), 0.8, 1.0)
    )
    
    # Regime persistence weighting
    regime_strength = (
        df['volume_regime_persistence'] * df['volatility_regime_persistence']
    )
    
    # Momentum efficiency regime filter
    efficiency_threshold = np.where(
        df['volatility_regime'] == 1, 0.4,  # higher threshold in high vol
        np.where(df['volatility_regime'] == -1, 0.2, 0.3)  # lower in low vol
    )
    
    df['efficient_momentum'] = np.where(
        df['momentum_efficiency'] > efficiency_threshold,
        df['base_momentum_signal'], 0
    )
    
    # Volume divergence detection
    df['volume_momentum_divergence'] = (
        np.sign(df['momentum_5d']) * df['volume_acceleration']
    )
    
    # Dynamic Alpha Factor Construction
    # Core factor calculation
    df['regime_adaptive_momentum'] = (
        df['efficient_momentum'] * 
        volatility_adjustment * 
        volume_confirmation * 
        (0.5 + 0.5 * regime_strength)
    )
    
    # Apply volume divergence filter
    divergence_penalty = np.where(
        df['volume_momentum_divergence'] < -0.2, 0.5,  # strong negative divergence
        np.where(df['volume_momentum_divergence'] < 0, 0.8, 1.0)  # mild negative divergence
    )
    
    # Final alpha factor with momentum persistence enhancement
    df['alpha_factor'] = (
        df['regime_adaptive_momentum'] * 
        divergence_penalty * 
        (0.3 + 0.7 * df['momentum_persistence_ratio'].clip(0, 2))
    )
    
    # Risk-adjusted signal validation
    # Filter out signals with poor regime alignment
    regime_alignment = (
        (df['volume_regime_persistence'] > 0.6) & 
        (df['volatility_regime_persistence'] > 0.6)
    )
    
    df['final_alpha'] = np.where(
        regime_alignment, 
        df['alpha_factor'], 
        df['alpha_factor'] * 0.5  # reduce signal strength for unstable regimes
    )
    
    # Normalize the final factor
    alpha_factor = df['final_alpha'].fillna(0)
    
    return alpha_factor
