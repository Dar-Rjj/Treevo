import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Alpha factor combining asymmetric momentum, volatility normalization, 
    volume regime detection, and intraday strength with regime-aware weighting.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Asymmetric Momentum Calculation
    momentum_5d = close / close.shift(5) - 1
    momentum_13d = close / close.shift(13) - 1
    momentum_34d = close / close.shift(34) - 1
    
    # Volatility Estimation Using Daily Range
    daily_range = (high - low) / close.shift(1)
    
    volatility_3d = daily_range.rolling(window=3).std()
    volatility_8d = daily_range.rolling(window=8).std()
    volatility_21d = daily_range.rolling(window=21).std()
    
    # Volatility-normalized momentum
    norm_momentum_short = momentum_5d / volatility_3d.replace(0, np.nan)
    norm_momentum_medium = momentum_13d / volatility_8d.replace(0, np.nan)
    norm_momentum_long = momentum_34d / volatility_21d.replace(0, np.nan)
    
    # Volume Regime Detection
    volume_rank = volume.rolling(window=20).apply(
        lambda x: (pd.Series(x).rank().iloc[-1] - 1) / (len(x) - 1) if len(x) == 20 else np.nan
    )
    volume_percentile = np.clip(volume_rank ** (1/3), 0, 1)
    
    volume_momentum = np.tanh(volume / volume.shift(5) - 1)
    
    # Intraday Strength Indicator
    close_position = (close - low) / (high - low).replace(0, np.nan)
    avg_close_position = close_position.rolling(window=3).mean()
    intraday_factor = 0.5 + 0.5 * np.tanh(2 * (avg_close_position - 0.5))
    
    # Factor Combination with Regime Weighting
    momentum_product = norm_momentum_short * norm_momentum_medium * norm_momentum_long
    momentum_signal = np.sign(momentum_product) * np.abs(momentum_product) ** (1/3)
    
    volume_strength = volume_percentile ** (1/3) * (1 + np.tanh(volume_momentum))
    
    adjusted_signal = momentum_signal * volume_strength * intraday_factor
    
    # Alternative Construction Path - Dynamic Weighting
    volume_weight = volume_percentile ** (1/2)
    short_weight = volume_weight
    long_weight = 1 - volume_weight
    
    weighted_momentum = (short_weight * norm_momentum_short + 
                        (1 - short_weight - long_weight) * norm_momentum_medium + 
                        long_weight * norm_momentum_long)
    
    # Final combination with regime awareness
    high_volume_regime = volume_percentile > 0.7
    low_volume_regime = volume_percentile < 0.3
    
    final_factor = pd.Series(index=df.index, dtype=float)
    final_factor[high_volume_regime] = adjusted_signal[high_volume_regime] * 0.7 + weighted_momentum[high_volume_regime] * 0.3
    final_factor[low_volume_regime] = adjusted_signal[low_volume_regime] * 0.3 + weighted_momentum[low_volume_regime] * 0.7
    final_factor[~high_volume_regime & ~low_volume_regime] = adjusted_signal[~high_volume_regime & ~low_volume_regime] * 0.5 + weighted_momentum[~high_volume_regime & ~low_volume_regime] * 0.5
    
    # Final bounded output
    final_factor = np.tanh(final_factor)
    
    return final_factor
