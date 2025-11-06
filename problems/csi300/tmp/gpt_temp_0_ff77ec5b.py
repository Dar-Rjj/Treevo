import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Multi-timeframe Momentum Alignment
    # Short-term momentum (1-3 days)
    ret_1d = df['close'] / df['close'].shift(1) - 1
    ret_3d = df['close'] / df['close'].shift(3) - 1
    short_momentum = 0.7 * ret_1d + 0.3 * ret_3d
    
    # Medium-term momentum (5-10 days)
    ret_5d = df['close'] / df['close'].shift(5) - 1
    ret_10d = df['close'] / df['close'].shift(10) - 1
    medium_momentum = 0.8 * ret_5d + 0.2 * ret_10d
    
    # Momentum alignment score
    short_rolling = short_momentum.rolling(window=5, min_periods=3).mean()
    medium_rolling = medium_momentum.rolling(window=5, min_periods=3).mean()
    sign_agreement = np.sign(short_rolling) * np.sign(medium_rolling)
    magnitude_similarity = 1 - np.abs(short_rolling - medium_rolling) / (np.abs(short_rolling) + np.abs(medium_rolling) + 1e-8)
    momentum_alignment = sign_agreement * magnitude_similarity
    
    # Volume Acceleration Confirmation
    # Volume trend calculation
    def calc_volume_slope(volume_series):
        slopes = []
        for i in range(len(volume_series)):
            if i >= 4:
                window = volume_series.iloc[i-4:i+1].values
                if len(window) == 5:
                    x = np.arange(5)
                    slope, _, _, _, _ = linregress(x, window)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=volume_series.index)
    
    volume_slope = calc_volume_slope(df['volume'])
    volume_acceleration = volume_slope.diff()
    avg_volume = df['volume'].rolling(window=20, min_periods=10).mean()
    normalized_acceleration = volume_acceleration / (avg_volume + 1e-8)
    
    # Volume-momentum synergy
    volume_synergy = momentum_alignment * normalized_acceleration
    
    # Volatility-Regime Adaptive Scaling
    # Regime detection
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    avg_range_20d = daily_range.rolling(window=20, min_periods=10).mean()
    
    high_vol_regime = daily_range > 1.5 * avg_range_20d
    low_vol_regime = daily_range < 0.7 * avg_range_20d
    normal_regime = ~high_vol_regime & ~low_vol_regime
    
    # Regime-aware volatility adjustment
    volatility_scaling = pd.Series(1.0, index=df.index)
    volatility_scaling[high_vol_regime] = 1 / (daily_range[high_vol_regime] + 1e-8)
    volatility_scaling[low_vol_regime] = 1.0
    
    # Regime persistence factor
    regime_persistence = high_vol_regime.rolling(window=5).sum() / 5
    volatility_scaling = volatility_scaling * (1 + 0.5 * regime_persistence)
    
    # Composite Alpha Generation
    # Combine aligned momentum with volume confirmation
    base_factor = momentum_alignment * volume_synergy
    
    # Apply regime-specific volatility scaling
    scaled_factor = base_factor * volatility_scaling
    
    # Multi-timeframe robustness check
    factor_1d = scaled_factor.rolling(window=1).mean()
    factor_3d = scaled_factor.rolling(window=3).mean()
    factor_5d = scaled_factor.rolling(window=5).mean()
    
    consistency_penalty = 1 - (np.abs(factor_1d - factor_3d) + np.abs(factor_1d - factor_5d)) / (np.abs(factor_1d) + 1e-8)
    consistency_penalty = np.clip(consistency_penalty, 0.5, 1.0)
    
    # Final alpha output
    final_alpha = scaled_factor * consistency_penalty
    
    return final_alpha
