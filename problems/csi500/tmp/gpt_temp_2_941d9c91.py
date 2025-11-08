import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-scaled momentum acceleration,
    volume-price divergence, and multi-timeframe signal integration.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Volatility-Scaled Momentum Acceleration
    # Multi-Timeframe Momentum
    mom_3d = (close - close.shift(3)) / close.shift(3)
    mom_5d = (close - close.shift(5)) / close.shift(5)
    mom_10d = (close - close.shift(10)) / close.shift(10)
    
    # Momentum Acceleration
    acc_3d = mom_3d - mom_5d
    acc_5d = mom_5d - mom_10d
    combined_acc = (acc_3d + acc_5d) / 2
    
    # Volatility Scaling
    vol_3d = close.pct_change(periods=1).rolling(window=3).std()
    vol_5d = close.pct_change(periods=1).rolling(window=5).std()
    vol_10d = close.pct_change(periods=1).rolling(window=10).std()
    
    # Combined Signal for Momentum
    vol_scaled_mom = np.cbrt(
        (mom_3d / vol_3d.replace(0, np.nan)) * 
        (mom_5d / vol_5d.replace(0, np.nan)) * 
        (mom_10d / vol_10d.replace(0, np.nan))
    )
    momentum_component = vol_scaled_mom * combined_acc
    
    # Volume-Price Divergence
    # Price Regime Detection
    trend_regime = np.sign(close - close.shift(5)) * np.sign(close - close.shift(10))
    vol_regime = (high - low) / ((high - low).rolling(window=5).mean())
    regime_score = trend_regime * vol_regime
    
    # Volume Divergence Components
    vol_ratio_3d = volume / volume.rolling(window=3).mean()
    vol_ratio_5d = volume / volume.rolling(window=5).mean()
    vol_ratio_10d = volume / volume.rolling(window=10).mean()
    
    # Price-Volume Alignment
    direction_align = np.sign(close - close.shift(1)) * np.sign(volume - volume.shift(1))
    strength_align = ((close - close.shift(1)) / close.shift(1)) * \
                    ((volume - volume.rolling(window=5).mean()) / volume.rolling(window=5).mean())
    
    # Persistence alignment
    def count_consecutive_direction(s):
        return s.rolling(window=5).apply(lambda x: (x == x.iloc[0]).sum() if len(x) == 5 else np.nan)
    
    persistence_align = count_consecutive_direction(direction_align)
    
    # Combined Signal for Volume Divergence
    geometric_vol_div = np.cbrt(vol_ratio_3d * vol_ratio_5d * vol_ratio_10d)
    volume_component = geometric_vol_div * regime_score * direction_align * strength_align * persistence_align
    
    # Multi-Timeframe Signal Integration
    # Short-term Framework
    short_price_mom = (close - close.shift(2)) / close.shift(2)
    short_vol_acc = volume / volume.shift(2)
    range_efficiency = (close - close.shift(1)) / (high - low).replace(0, np.nan)
    
    # Medium-term Framework
    medium_price_trend = (close - close.shift(7)) / close.shift(7)
    medium_vol_trend = volume / volume.rolling(window=7).mean()
    vol_persistence = (high - low) / ((high - low).rolling(window=7).mean())
    
    # Combined Signal for Multi-Timeframe
    signal_convergence = short_price_mom * medium_price_trend
    vol_geo_align = np.cbrt(short_vol_acc * medium_vol_trend * (volume / volume.shift(1)))
    timeframe_component = signal_convergence * vol_geo_align * regime_score
    
    # Final Alpha Factor
    alpha_factor = np.cbrt(
        momentum_component.replace([np.inf, -np.inf], np.nan) * 
        volume_component.replace([np.inf, -np.inf], np.nan) * 
        timeframe_component.replace([np.inf, -np.inf], np.nan)
    )
    
    return alpha_factor
