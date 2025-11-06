import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics with Regime Transition Detection
    """
    data = df.copy()
    
    # Fractal Price Pattern Analysis
    # Multi-scale price range fractality
    data['short_term_compression'] = (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3))
    data['medium_term_expansion'] = (data['high'].shift(5) - data['low'].shift(5)) / (data['high'].shift(10) - data['low'].shift(10))
    
    # Fractal similarity score across 3 time windows
    data['range_1'] = data['high'] - data['low']
    data['range_3'] = data['high'].shift(2) - data['low'].shift(2)
    data['range_5'] = data['high'].shift(4) - data['low'].shift(4)
    data['fractal_similarity'] = (data['range_1'].rolling(window=3).std() + 1e-8) / \
                                (data['range_1'].rolling(window=3).mean() + 1e-8)
    
    # Price level clustering detection
    data['support_resistance_density'] = data['close'].rolling(window=20).apply(
        lambda x: np.sum(np.abs((x.iloc[-1] - x.iloc[:-1]) / x.iloc[:-1]) <= 0.02)
    )
    
    # Breakout intensity
    data['prev_5d_high'] = data['high'].rolling(window=5).apply(lambda x: x.iloc[:-1].max())
    data['breakout_intensity'] = (data['close'] - data['prev_5d_high']) / (data['high'] - data['low'] + 1e-8)
    
    # Consolidation phase identification
    data['volatility_20d'] = data['close'].rolling(window=20).std()
    data['consolidation_score'] = (data['volatility_20d'].shift(1) / (data['volatility_20d'] + 1e-8)) * \
                                 data['support_resistance_density']
    
    # Volume Fractal Structure
    # Volume distribution self-similarity
    data['volume_persistence'] = data['volume'] / (data['volume'].rolling(window=5).mean() + 1e-8)
    
    # Multi-day volume clustering
    data['volume_clustering'] = data['volume'].rolling(window=8).apply(
        lambda x: np.sum(x.iloc[:-1] > 1.2 * x.iloc[-1])
    )
    
    # Volume burst sequencing
    data['volume_ratio'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    data['volume_burst_pattern'] = data['volume_ratio'].rolling(window=5).apply(
        lambda x: np.sum((x > 1.5) | (x < 0.67))
    )
    
    # Volume-price fractal alignment
    data['range_volume_alignment'] = ((data['high'] - data['low']) / data['high'].rolling(window=5).mean()) * \
                                    data['volume_persistence']
    
    # Regime Transition Probability
    # Volatility regime shift indicators
    data['avg_range_5d'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['range_expansion_onset'] = (data['high'] - data['low']) / (data['avg_range_5d'] + 1e-8)
    
    # Volatility clustering persistence
    data['range_magnitude'] = data['high'] - data['low']
    data['volatility_clustering'] = data['range_magnitude'].rolling(window=5).apply(
        lambda x: np.sum(np.abs((x.iloc[:-1] - x.iloc[-1]) / x.iloc[-1]) <= 0.15)
    )
    
    # Breakout confirmation strength
    data['breakout_confirmation'] = data['range_expansion_onset'] * data['volume_persistence']
    
    # Momentum regime transition
    # Acceleration/deceleration phase detection
    data['price_velocity_short'] = data['close'] - data['close'].shift(2)
    data['price_velocity_medium'] = data['close'].shift(2) - data['close'].shift(4)
    data['price_acceleration'] = data['price_velocity_short'] - data['price_velocity_medium']
    
    data['volume_acceleration_short'] = data['volume'] / (data['volume'].shift(2) + 1e-8)
    data['volume_acceleration_medium'] = data['volume'].shift(2) / (data['volume'].shift(4) + 1e-8)
    data['volume_acceleration'] = data['volume_acceleration_short'] - data['volume_acceleration_medium']
    
    # Trend exhaustion signals
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['trend_strength'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['price_range_divergence'] = data['close_position'] * np.abs(data['trend_strength'])
    
    # Volume drying during trend persistence
    data['volume_trend_divergence'] = data['trend_strength'] / (data['volume_persistence'] + 1e-8)
    
    # Mean reversion regime probability
    data['extreme_clustering'] = data['support_resistance_density'] * \
                                (1 / (np.abs(data['close'] / data['close'].rolling(window=20).mean() - 1) + 0.1))
    
    # Fractal Timing Signals
    # Multi-timeframe pattern alignment
    data['short_medium_correlation'] = data['short_term_compression'].rolling(window=10).corr(
        data['medium_term_expansion']
    )
    
    # Volume pattern synchronization
    data['volume_sync'] = data['volume_persistence'].rolling(window=5).std() / \
                         (data['volume_persistence'].rolling(window=5).mean() + 1e-8)
    
    # Pattern completion probability
    data['range_compression_trend'] = data['short_term_compression'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )
    
    # Volume clustering trend
    data['volume_clustering_trend'] = data['volume_clustering'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )
    
    # Timing based on fractal maturity
    data['pattern_duration'] = data['consolidation_score'].rolling(window=10).apply(
        lambda x: len(x) - np.argmax(x)
    )
    
    # Transition readiness scoring
    data['regime_convergence'] = (
        data['range_expansion_onset'].rolling(window=5).mean() +
        data['price_acceleration'].rolling(window=5).mean() +
        data['volume_acceleration'].rolling(window=5).mean()
    ) / 3
    
    data['fractal_clarity'] = 1 / (data['fractal_similarity'] + 1e-8)
    data['volume_price_alignment'] = data['range_volume_alignment'].rolling(window=5).mean()
    
    # Final factor construction with risk-adjusted combination
    data['fractal_regime_score'] = (
        0.15 * data['range_expansion_onset'] +
        0.12 * data['price_acceleration'] +
        0.10 * data['volume_acceleration'] +
        0.13 * data['breakout_confirmation'] +
        0.11 * data['extreme_clustering'] +
        0.09 * data['short_medium_correlation'].fillna(0) +
        0.08 * data['regime_convergence'] +
        0.07 * data['fractal_clarity'] +
        0.08 * data['volume_price_alignment'] +
        0.07 * data['pattern_duration'] / data['pattern_duration'].rolling(window=20).max()
    )
    
    # Apply regime-dependent weighting
    volatility_regime = data['range_expansion_onset'] > 1.2
    momentum_regime = np.abs(data['price_acceleration']) > 0.02
    mean_reversion_regime = data['extreme_clustering'] > data['extreme_clustering'].rolling(window=20).quantile(0.7)
    
    data['regime_weight'] = (
        0.4 * volatility_regime.astype(int) +
        0.35 * momentum_regime.astype(int) +
        0.25 * mean_reversion_regime.astype(int)
    )
    
    # Final factor with regime adjustment
    factor = data['fractal_regime_score'] * data['regime_weight']
    
    return factor
