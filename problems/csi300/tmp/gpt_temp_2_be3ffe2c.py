import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Entropy Dynamics factor
    """
    df = df.copy()
    
    # Price Fractal Entropy
    # Short-term price entropy
    price_ratios_short = pd.DataFrame({
        'r1': df['close'] / df['close'].shift(1),
        'r2': df['close'].shift(1) / df['close'].shift(2),
        'r3': df['close'].shift(2) / df['close'].shift(3)
    })
    price_entropy_short = price_ratios_short.std(axis=1)
    
    # Medium-term price entropy
    price_ratios_medium = pd.DataFrame({
        'r1': df['close'] / df['close'].shift(3),
        'r2': df['close'].shift(3) / df['close'].shift(6),
        'r3': df['close'].shift(6) / df['close'].shift(9)
    })
    price_entropy_medium = price_ratios_medium.std(axis=1)
    
    # Long-term price entropy
    price_ratios_long = pd.DataFrame({
        'r1': df['close'] / df['close'].shift(6),
        'r2': df['close'].shift(6) / df['close'].shift(12),
        'r3': df['close'].shift(12) / df['close'].shift(18)
    })
    price_entropy_long = price_ratios_long.std(axis=1)
    
    # Volume Fractal Entropy
    # Short-term volume entropy
    volume_ratios_short = pd.DataFrame({
        'r1': df['volume'] / df['volume'].shift(1),
        'r2': df['volume'].shift(1) / df['volume'].shift(2),
        'r3': df['volume'].shift(2) / df['volume'].shift(3)
    })
    volume_entropy_short = volume_ratios_short.std(axis=1)
    
    # Medium-term volume entropy
    volume_ratios_medium = pd.DataFrame({
        'r1': df['volume'] / df['volume'].shift(3),
        'r2': df['volume'].shift(3) / df['volume'].shift(6),
        'r3': df['volume'].shift(6) / df['volume'].shift(9)
    })
    volume_entropy_medium = volume_ratios_medium.std(axis=1)
    
    # Long-term volume entropy
    volume_ratios_long = pd.DataFrame({
        'r1': df['volume'] / df['volume'].shift(6),
        'r2': df['volume'].shift(6) / df['volume'].shift(12),
        'r3': df['volume'].shift(12) / df['volume'].shift(18)
    })
    volume_entropy_long = volume_ratios_long.std(axis=1)
    
    # Combined multi-scale entropy
    combined_price_entropy = (price_entropy_short + price_entropy_medium + price_entropy_long) / 3
    combined_volume_entropy = (volume_entropy_short + volume_entropy_medium + volume_entropy_long) / 3
    
    # Entropy Regime Classification
    price_entropy_threshold = combined_price_entropy.rolling(window=20, min_periods=10).quantile(0.6)
    volume_entropy_threshold = combined_volume_entropy.rolling(window=20, min_periods=10).quantile(0.6)
    
    high_entropy_regime = (combined_price_entropy > price_entropy_threshold) & (combined_volume_entropy > volume_entropy_threshold)
    low_entropy_regime = (combined_price_entropy <= price_entropy_threshold) & (combined_volume_entropy <= volume_entropy_threshold)
    mixed_entropy_regime = ~(high_entropy_regime | low_entropy_regime)
    
    # Fractal Momentum Asymmetry
    # Upside Momentum Concentration
    positive_returns = pd.DataFrame({
        'r1': (df['close'] / df['close'].shift(1)) > 1,
        'r2': (df['close'] / df['close'].shift(2)) > 1,
        'r3': (df['close'] / df['close'].shift(3)) > 1
    })
    positive_return_count = positive_returns.sum(axis=1)
    
    # Consecutive positive returns
    returns = df['close'] / df['close'].shift(1) - 1
    consecutive_pos = returns.rolling(window=5).apply(
        lambda x: max(len(list(g)) for k, g in groupby(x > 0) if k) if any(x > 0) else 0, 
        raw=False
    )
    
    # Downside Momentum Dispersion
    negative_returns = returns[returns < 0]
    negative_return_std = negative_returns.rolling(window=10, min_periods=5).std()
    
    # Volume-Price Phase Divergence
    price_entropy_change = combined_price_entropy.diff(3)
    volume_entropy_change = combined_volume_entropy.diff(3)
    
    price_up_volume_down = (price_entropy_change > 0) & (volume_entropy_change < 0)
    volume_up_price_down = (volume_entropy_change > 0) & (price_entropy_change < 0)
    synchronized_entropy = (price_entropy_change * volume_entropy_change) > 0
    
    # Smart Money Entropy Detection
    amount_rolling = df['amount'].rolling(window=10, min_periods=5).mean()
    high_amount_low_entropy = (df['amount'] > amount_rolling) & (combined_price_entropy < price_entropy_threshold)
    low_amount_high_entropy = (df['amount'] < amount_rolling) & (combined_price_entropy > price_entropy_threshold)
    
    volume_spike = df['volume'] > df['volume'].rolling(window=10, min_periods=5).mean() * 1.5
    volume_spike_price_stable = volume_spike & (combined_price_entropy < price_entropy_threshold)
    
    # Multi-Fractal Regime Integration
    # Short-term momentum
    short_momentum = df['close'] / df['close'].shift(3) - 1
    
    # Medium-term momentum
    medium_momentum = df['close'] / df['close'].shift(6) - 1
    
    # Long-term momentum
    long_momentum = df['close'] / df['close'].shift(12) - 1
    
    # High Entropy Momentum Factor
    high_entropy_momentum = (
        short_momentum * volume_entropy_short.rolling(window=5).mean() +
        medium_momentum * (1 - price_entropy_medium.rolling(window=5).mean()) +
        long_momentum * ((combined_price_entropy + combined_volume_entropy) / 2).rolling(window=5).mean()
    )
    
    # Low Entropy Mean Reversion Factor
    price_range = df['high'] - df['low']
    price_to_range = df['close'] / price_range.replace(0, np.nan)
    
    volume_avg_4 = df['volume'].rolling(window=4, min_periods=2).mean()
    volume_compression = df['volume'] / volume_avg_4.replace(0, np.nan)
    
    mean_reversion_strength = (1 - combined_price_entropy.rolling(window=5).mean()) * price_to_range
    
    low_entropy_mean_reversion = (
        price_to_range.rolling(window=5).mean() +
        (1 - volume_compression.rolling(window=5).mean()) +
        mean_reversion_strength.rolling(window=5).mean()
    )
    
    # Dynamic Entropy Transition Signals
    entropy_increase = combined_price_entropy / combined_price_entropy.shift(3) > 1.2
    entropy_slope = combined_price_entropy.rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    entropy_decay = entropy_slope < 0
    
    entropy_crossover = (combined_price_entropy - combined_volume_entropy).diff(3)
    
    # Adaptive Factor Synthesis
    # Regime-dependent weighting
    regime_weight = pd.Series(0.0, index=df.index)
    regime_weight[high_entropy_regime] = 0.7
    regime_weight[low_entropy_regime] = 0.3
    regime_weight[mixed_entropy_regime] = 0.5
    
    # Multi-scale entropy divergence
    entropy_divergence = (
        (price_entropy_short - volume_entropy_short).abs() +
        (price_entropy_medium - volume_entropy_medium).abs() +
        (price_entropy_long - volume_entropy_long).abs()
    ) / 3
    
    # Final fractal entropy momentum factor
    fractal_entropy_factor = (
        regime_weight * high_entropy_momentum.rolling(window=3).mean() +
        (1 - regime_weight) * low_entropy_mean_reversion.rolling(window=3).mean() +
        entropy_divergence.rolling(window=5).mean() * positive_return_count.rolling(window=3).mean() -
        negative_return_std.rolling(window=5).mean() * 0.5
    )
    
    # Normalize the final factor
    fractal_entropy_factor = (fractal_entropy_factor - fractal_entropy_factor.rolling(window=20, min_periods=10).mean()) / fractal_entropy_factor.rolling(window=20, min_periods=10).std()
    
    return fractal_entropy_factor

def groupby(iterable):
    """Helper function for consecutive counting"""
    from itertools import groupby as itertools_groupby
    return [(k, list(g)) for k, g in itertools_groupby(iterable)]
