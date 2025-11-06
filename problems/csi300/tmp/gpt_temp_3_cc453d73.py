import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Adaptive Momentum Persistence with Volume-Price Divergence Detection
    Generates alpha factor combining momentum, volume, persistence, and range analysis
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Framework
    df['momentum_2d'] = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    df['momentum_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['momentum_convergence'] = df['momentum_2d'] - df['momentum_5d']
    
    # Adaptive Volume Confirmation System
    df['volume_ma_3'] = df['volume'].rolling(window=3).mean()
    df['volume_trend'] = df['volume'] / df['volume_ma_3']
    
    # Volume-Price Divergence Detection
    volume_momentum_sign = np.sign(df['volume_trend']) * np.sign(df['momentum_5d'])
    df['volume_confidence'] = np.where(
        volume_momentum_sign > 0, 1.0,  # High confidence: same direction
        np.where(
            volume_momentum_sign == 0, 0.7,  # Medium confidence: neutral
            0.3  # Low confidence: divergence
        )
    )
    
    # Dynamic Persistence Scoring
    # Direction Persistence Strength
    momentum_direction = np.sign(df['momentum_5d'])
    persistence_count = (momentum_direction == momentum_direction.shift(1)).astype(int)
    persistence_weights = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i == 0:
            persistence_weights.iloc[i] = 0
            continue
            
        count = 0
        j = i
        while j >= 0 and momentum_direction.iloc[j] == momentum_direction.iloc[i]:
            count += 1
            j -= 1
            if j < 0:
                break
        
        # Exponential weighting: sum(2^(-i) for i=0 to count-1)
        weight_sum = sum(2**(-k) for k in range(count))
        persistence_weights.iloc[i] = weight_sum
    
    # Magnitude Stability Assessment
    df['momentum_2d_std_4d'] = df['momentum_2d'].rolling(window=4).std()
    df['magnitude_stability'] = 1 / (df['momentum_2d_std_4d'] + 0.0001)
    
    # Combined Persistence Metric
    df['persistence_metric'] = (persistence_weights * df['magnitude_stability'] * 
                               np.abs(df['momentum_5d']))
    
    # Price Range Context Analysis
    df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
    df['relative_range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Range-Based Signal Adjustment
    range_adjustment = np.where(
        (df['momentum_5d'] > 0) & (df['relative_range_position'] > 0.7), 1.2,  # Strong uptrend
        np.where(
            (df['momentum_5d'] < 0) & (df['relative_range_position'] < 0.3), 1.2,  # Strong downtrend
            np.where(
                (df['momentum_5d'] > 0) & (df['relative_range_position'] < 0.3), 0.8,  # Weak uptrend
                np.where(
                    (df['momentum_5d'] < 0) & (df['relative_range_position'] > 0.7), 0.8,  # Weak downtrend
                    1.0  # Neutral
                )
            )
        )
    )
    
    # Regime-Adaptive Weighting
    # Volatility Regime Detection
    df['returns_10d_std'] = df['close'].pct_change().rolling(window=10).std()
    df['volatility_20d_avg'] = df['close'].pct_change().rolling(window=20).std()
    high_vol_regime = df['returns_10d_std'] > df['volatility_20d_avg']
    
    # Momentum Regime Assessment
    df['momentum_autocorr'] = df['momentum_5d'].rolling(window=10).apply(
        lambda x: x.autocorr() if len(x) == 10 else np.nan, raw=False
    )
    trend_regime = df['momentum_autocorr'] > 0.3
    
    # Final Alpha Generation
    # Core Momentum Signal
    core_momentum = df['momentum_5d'] * (1 + 0.5 * df['momentum_convergence'])
    
    # Volume-Adapted Signal
    volume_adapted = core_momentum * df['volume_confidence']
    
    # Persistence-Enhanced Signal
    persistence_enhanced = volume_adapted * df['persistence_metric']
    
    # Range-Optimized Signal
    range_optimized = (persistence_enhanced * range_adjustment) / (df['intraday_volatility'] + 0.0001)
    
    # Regime-Adaptive Final Alpha
    final_alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if pd.isna(high_vol_regime.iloc[i]) or pd.isna(trend_regime.iloc[i]):
            final_alpha.iloc[i] = range_optimized.iloc[i]
            continue
            
        if high_vol_regime.iloc[i]:
            # High volatility regime
            if trend_regime.iloc[i]:
                # High volatility, trend regime
                weights = [0.4, 0.3, 0.2, 0.1]  # persistence, range, momentum, volume
            else:
                # High volatility, mean-reversion regime
                weights = [0.1, 0.3, 0.2, 0.4]  # persistence, range, momentum, volume
        else:
            # Low volatility regime
            if trend_regime.iloc[i]:
                # Low volatility, trend regime
                weights = [0.15, 0.0, 0.35, 0.3]  # persistence, range, momentum, convergence
            else:
                # Low volatility, mean-reversion regime
                weights = [0.1, 0.3, 0.2, 0.4]  # persistence, range, momentum, volume
        
        if high_vol_regime.iloc[i]:
            # High volatility weighting
            components = [
                df['persistence_metric'].iloc[i],
                range_adjustment[i],
                core_momentum.iloc[i],
                df['volume_confidence'].iloc[i]
            ]
        else:
            # Low volatility weighting
            components = [
                df['persistence_metric'].iloc[i],
                0,  # range not used in low volatility
                core_momentum.iloc[i],
                df['momentum_convergence'].iloc[i]
            ]
        
        final_alpha.iloc[i] = sum(w * c for w, c in zip(weights, components))
    
    return final_alpha
