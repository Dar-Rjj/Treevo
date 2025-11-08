import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Classification
    close_std_10 = df['close'].rolling(window=10, min_periods=10).std()
    close_std_20 = df['close'].rolling(window=20, min_periods=20).std()
    volatility_ratio = close_std_10 / close_std_20
    
    # Regime States
    high_vol_regime = volatility_ratio > 1.2
    low_vol_regime = volatility_ratio < 0.8
    normal_vol_regime = ~(high_vol_regime | low_vol_regime)
    
    # Pressure Accumulation Measurement
    intraday_pressure = (df['close'] - df['open']) / (df['high'] - df['low'])
    intraday_pressure = intraday_pressure.replace([np.inf, -np.inf], np.nan)
    
    cumulative_pressure = intraday_pressure.rolling(window=5, min_periods=5).sum()
    
    volume_weighted_pressure_numerator = (intraday_pressure * df['volume']).rolling(window=5, min_periods=5).sum()
    volume_weighted_pressure_denominator = df['volume'].rolling(window=5, min_periods=5).sum()
    volume_weighted_pressure = volume_weighted_pressure_numerator / volume_weighted_pressure_denominator
    
    # Volume Pattern Analysis
    volume_trend = df['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: np.corrcoef(range(1, 6), x.values)[0, 1] if len(x) == 5 else np.nan, 
        raw=False
    )
    
    volume_mean_20 = df['volume'].rolling(window=20, min_periods=20).mean()
    volume_clustering = df['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: (x > volume_mean_20.loc[x.index]).sum() / 5, 
        raw=False
    )
    
    volume_mean_5 = df['volume'].rolling(window=5, min_periods=5).mean()
    volume_exhaustion = df['volume'] / volume_mean_5 - 1
    
    # Composite Signal Generation
    # Regime-adaptive pressure core
    regime_pressure_core = pd.Series(index=df.index, dtype=float)
    regime_pressure_core[high_vol_regime] = cumulative_pressure[high_vol_regime] * volatility_ratio[high_vol_regime]
    regime_pressure_core[low_vol_regime] = cumulative_pressure[low_vol_regime] / volatility_ratio[low_vol_regime]
    regime_pressure_core[normal_vol_regime] = cumulative_pressure[normal_vol_regime]
    
    # Volume confirmation
    volume_confirmation = (volume_trend * volume_weighted_pressure) * (volume_clustering * volume_exhaustion)
    
    # Recent price efficiency
    recent_price_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    recent_price_efficiency = recent_price_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Final alpha
    alpha = regime_pressure_core * volume_confirmation * recent_price_efficiency
    
    return alpha
