import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Effective spread calculation
    mid_price = (df['high'] + df['low']) / 2
    effective_spread = 2 * np.abs(mid_price - df['close']) / mid_price
    
    # Spread momentum and volatility
    spread_momentum = effective_spread - effective_spread.shift(5)
    spread_volatility = effective_spread.rolling(window=10, min_periods=5).std()
    
    # Volume microstructure
    large_trade_concentration = df['amount'] / (df['volume'] * df['close'])
    daily_volume_concentration = df['volume'] / df['volume'].rolling(window=5, min_periods=3).mean()
    volume_clustering = daily_volume_concentration.rolling(window=5, min_periods=3).std()
    microstructure_noise = volume_clustering / (spread_volatility + 1e-8)
    
    # Price impact components
    permanent_impact = df['close'].rolling(window=10, min_periods=5).corr(df['volume'] * (df['close'] - df['open']))
    transient_impact = (df['close'] - df['open']) / (np.sqrt(df['volume']) + 1e-8)
    impact_ratio = permanent_impact / (np.abs(transient_impact) + 1e-8)
    
    # Regime-dependent signals
    noise_threshold_high = microstructure_noise.quantile(0.7)
    noise_threshold_low = microstructure_noise.quantile(0.3)
    
    regime_signal = np.where(
        microstructure_noise > noise_threshold_high,
        impact_ratio * microstructure_noise,
        np.where(
            microstructure_noise < noise_threshold_low,
            impact_ratio / (microstructure_noise + 1e-8),
            impact_ratio
        )
    )
    
    # Order flow dynamics
    opening_pressure = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-8)
    closing_pressure = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    pressure_divergence = opening_pressure - closing_pressure
    
    volume_acceleration = df['volume'] / (df['volume'].shift(1) + 1e-8) - 1
    amount_acceleration = df['amount'] / (df['amount'].shift(1) + 1e-8) - 1
    flow_momentum = volume_acceleration * amount_acceleration
    
    # Regime consistency
    regime = pd.cut(microstructure_noise, bins=[-np.inf, noise_threshold_low, noise_threshold_high, np.inf], labels=[0, 1, 2])
    regime_persistence = regime.rolling(window=3, min_periods=2).apply(lambda x: len(set(x.dropna())) if len(x.dropna()) > 0 else np.nan)
    regime_confidence = 1 - (1 / (1 + regime_persistence))
    
    # Regime entropy calculation
    def calculate_entropy(series):
        counts = series.value_counts(normalize=True)
        return -np.sum(counts * np.log(counts + 1e-8))
    
    regime_entropy = regime.rolling(window=10, min_periods=5).apply(calculate_entropy, raw=False)
    regime_quality = 1 - (regime_entropy / np.log(3))
    
    # Alpha synthesis
    microstructure_signal = regime_signal
    flow_signal = pressure_divergence * flow_momentum
    quality_weight = regime_confidence * regime_quality
    
    raw_alpha = microstructure_signal * flow_signal
    final_alpha = raw_alpha * quality_weight
    
    return pd.Series(final_alpha, index=df.index)
