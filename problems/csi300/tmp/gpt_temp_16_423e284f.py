import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Entropy-Volume Divergence Factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Multi-Scale Entropy-Volume Dynamics
    # Volume-Price Entropy Divergence
    def calculate_entropy_divergence(short_lag, medium_lag, long_lag):
        # Short-term divergence
        short_div = (df['volume'] / df['volume'].shift(short_lag)) * \
                   ((df['high'] - df['low']) / (df['high'].shift(short_lag) - df['low'].shift(short_lag))) - \
                   (df['volume'].shift(short_lag) / df['volume'].shift(medium_lag)) * \
                   ((df['high'].shift(short_lag) - df['low'].shift(short_lag)) / 
                    (df['high'].shift(medium_lag) - df['low'].shift(medium_lag)))
        
        # Medium-term divergence
        medium_div = (df['volume'] / df['volume'].shift(medium_lag)) * \
                    ((df['high'] - df['low']) / (df['high'].shift(medium_lag) - df['low'].shift(medium_lag))) - \
                    (df['volume'].shift(medium_lag) / df['volume'].shift(long_lag)) * \
                    ((df['high'].shift(medium_lag) - df['low'].shift(medium_lag)) / 
                     (df['high'].shift(long_lag) - df['low'].shift(long_lag)))
        
        # Long-term divergence
        long_div = (df['volume'] / df['volume'].shift(long_lag)) * \
                  ((df['high'] - df['low']) / (df['high'].shift(long_lag) - df['low'].shift(long_lag))) - \
                  (df['volume'].shift(long_lag) / df['volume'].shift(long_lag*2)) * \
                  ((df['high'].shift(long_lag) - df['low'].shift(long_lag)) / 
                   (df['high'].shift(long_lag*2) - df['low'].shift(long_lag*2)))
        
        return short_div, medium_div, long_div
    
    # Calculate multi-scale divergences
    short_div_5d, medium_div_13d, long_div_34d = calculate_entropy_divergence(2, 5, 13)
    
    # Entropy Persistence Patterns
    def calculate_entropy_persistence(entropy_series, window):
        entropy_momentum = entropy_series.diff()
        entropy_consistency = entropy_series.rolling(window=window).apply(
            lambda x: np.sum(np.sign(x.diff().dropna())) if len(x.dropna()) >= window-1 else np.nan, raw=False
        )
        entropy_acceleration = (entropy_series / entropy_series.shift(3) - 1) * (df['high'] - df['low'])
        return entropy_momentum, entropy_consistency, entropy_acceleration
    
    short_ent_momentum, short_ent_consistency, short_ent_acceleration = calculate_entropy_persistence(short_div_5d, 5)
    medium_ent_momentum, medium_ent_consistency, medium_ent_acceleration = calculate_entropy_persistence(medium_div_13d, 13)
    long_ent_momentum, long_ent_consistency, long_ent_acceleration = calculate_entropy_persistence(long_div_34d, 34)
    
    # Volume Concentration Analysis
    volume_rolling = df['volume'].rolling(window=5)
    volume_median = volume_rolling.median()
    volume_max = volume_rolling.max()
    volume_min = volume_rolling.min()
    volume_sum = volume_rolling.sum()
    
    volume_skew = (df['volume'] - volume_median) / (volume_max - volume_min).replace(0, np.nan)
    concentration_ratio = df['volume'] / volume_sum
    volume_range = (df['volume'] - volume_min) / (volume_max - volume_min).replace(0, np.nan)
    
    # Microstructure Regime Classification
    # Bid-Ask Spread Proxy
    relative_spread = (df['high'] - df['low']) / df['close']
    
    def rolling_correlation(x, y, window):
        return x.rolling(window=window).corr(y)
    
    spread_persistence = rolling_correlation(df['high'] - df['low'], df['volume'], 5)
    
    # Entropy Regime Detection
    def detect_entropy_regime(short_ent, medium_ent, long_ent):
        # Calculate entropy volatility
        short_vol = short_ent.rolling(window=5).std()
        medium_vol = medium_ent.rolling(window=13).std()
        long_vol = long_ent.rolling(window=34).std()
        
        # Regime classification
        low_entropy = (short_vol < short_vol.quantile(0.33)) & \
                     (medium_vol < medium_vol.quantile(0.33)) & \
                     (long_vol < long_vol.quantile(0.33))
        
        high_entropy = (short_vol > short_vol.quantile(0.67)) | \
                      (medium_vol > medium_vol.quantile(0.67)) | \
                      (long_vol > long_vol.quantile(0.67))
        
        medium_entropy = ~low_entropy & ~high_entropy
        
        return low_entropy, medium_entropy, high_entropy
    
    low_ent_regime, medium_ent_regime, high_ent_regime = detect_entropy_regime(
        short_div_5d, medium_div_13d, long_div_34d
    )
    
    # Regime Quality Assessment
    entropy_consistency = pd.concat([
        short_ent_momentum.rolling(window=3).std(),
        medium_ent_momentum.rolling(window=3).std(),
        long_ent_momentum.rolling(window=3).std()
    ], axis=1).mean(axis=1)
    
    volume_entropy_alignment = rolling_correlation(df['volume'], short_div_5d, 3)
    
    regime_quality = (1 / (1 + entropy_consistency)) * (1 + volume_entropy_alignment)
    
    # Price-Volume Asymmetry Integration
    # Directional Volume Pressure
    up_pressure = ((df['close'] - df['low']) * df['volume']) / (df['high'] - df['low']).replace(0, np.nan)
    down_pressure = ((df['high'] - df['close']) * df['volume']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Asymmetry Ratio Analysis
    short_asymmetry = up_pressure.rolling(window=3).sum() / down_pressure.rolling(window=3).sum().replace(0, np.nan)
    long_asymmetry = up_pressure.rolling(window=6).sum() / down_pressure.rolling(window=6).sum().replace(0, np.nan)
    
    # Asymmetry-Entropy Divergence
    short_asym_ent_div = short_asymmetry / short_div_5d.replace(0, np.nan)
    long_asym_ent_div = long_asymmetry / long_div_34d.replace(0, np.nan)
    
    # Breakout Pattern Recognition
    # Liquidity Compression Signals
    volume_compression = volume_range < 0.3
    price_efficiency = ((df['close'] - df['open']) ** 2) / ((df['high'] - df['low']) ** 2).replace(0, np.nan)
    entropy_compression = short_ent_momentum.rolling(window=3).std() < short_ent_momentum.rolling(window=20).quantile(0.2)
    
    # Breakout Intensity Classification
    strong_breakout = (volume_range > 0.8) & (price_efficiency > 0.7) & (short_ent_acceleration > 0)
    moderate_breakout = (volume_range > 0.6) & (price_efficiency > 0.5) & (short_ent_momentum > 0)
    weak_breakout = ~strong_breakout & ~moderate_breakout
    
    # Composite Alpha Generation
    # Core Divergence Signal
    core_divergence = (
        short_div_5d * 0.4 + 
        medium_div_13d * 0.35 + 
        long_div_34d * 0.25
    ) * regime_quality
    
    # Asymmetry-Entropy Alignment
    asym_ent_alignment = (short_asym_ent_div + long_asym_ent_div) / 2
    
    # Volume Concentration Enhancement
    volume_enhancement = concentration_ratio * volume_skew
    
    # Regime-Adaptive Weighting
    regime_weight = pd.Series(1.0, index=df.index)
    regime_weight[low_ent_regime] = 0.6  # Emphasize long-term
    regime_weight[medium_ent_regime] = 0.8  # Balanced weighting
    regime_weight[high_ent_regime] = 1.2  # Focus on short-term
    
    # Breakout Integration
    breakout_intensity = pd.Series(1.0, index=df.index)
    breakout_intensity[strong_breakout] = 1.5
    breakout_intensity[moderate_breakout] = 1.2
    breakout_intensity[weak_breakout] = 0.8
    
    # Final Alpha Factor
    alpha_factor = (
        core_divergence * 
        regime_weight * 
        breakout_intensity * 
        (1 + volume_enhancement.fillna(0))
    )
    
    return alpha_factor
