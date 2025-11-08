import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-scale Momentum Analysis
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_accel'] = data['momentum_5d'] - data['momentum_5d'].shift(5)
    
    # Volume Regime Classification
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Calculate up-day and down-day volume intensities
    up_days = data['close'] > data['close'].shift(1)
    down_days = data['close'] < data['close'].shift(1)
    
    data['up_day_volume_intensity'] = data['volume'].rolling(window=20).apply(
        lambda x: x[up_days.loc[x.index]].mean() if up_days.loc[x.index].any() else np.nan
    )
    data['down_day_volume_intensity'] = data['volume'].rolling(window=20).apply(
        lambda x: x[down_days.loc[x.index]].mean() if down_days.loc[x.index].any() else np.nan
    )
    
    data['volume_pressure_asymmetry'] = data['up_day_volume_intensity'] / data['down_day_volume_intensity']
    
    # Volatility Context
    data['returns'] = data['close'].pct_change()
    data['volatility_20d'] = data['returns'].rolling(window=20).std()
    
    # Avoid division by zero in range utilization
    high_low_range = data['high'] - data['low']
    data['range_utilization'] = np.where(high_low_range > 0, 
                                        abs(data['close'] - data['open']) / high_low_range, 
                                        0)
    
    # Volatility regime classification
    volatility_60d = data['volatility_20d'].rolling(window=60, min_periods=20).apply(
        lambda x: np.percentile(x, 80) if len(x) >= 20 else np.nan
    )
    data['volatility_regime_high'] = data['volatility_20d'] > volatility_60d
    data['volatility_regime_low'] = data['volatility_20d'] < data['volatility_20d'].rolling(
        window=60, min_periods=20).apply(lambda x: np.percentile(x, 20) if len(x) >= 20 else np.nan)
    
    # Regime Divergence Detection
    data['positive_divergence'] = (data['momentum_accel'] > 0) & (data['volume_momentum'] < 0)
    data['negative_divergence'] = (data['momentum_accel'] < 0) & (data['volume_momentum'] > 0)
    
    data['strong_directional_volume'] = (data['momentum_5d'] > 0) & (data['volume_pressure_asymmetry'] > 1.2)
    data['weak_directional_volume'] = (data['momentum_5d'] > 0) & (data['volume_pressure_asymmetry'] < 0.8)
    
    data['efficient_vol_momentum'] = (data['momentum_5d'] > 0) & data['volatility_regime_low'] & (data['range_utilization'] > 0.6)
    data['inefficient_vol_momentum'] = (data['momentum_5d'] > 0) & data['volatility_regime_high'] & (data['range_utilization'] < 0.4)
    
    # Alpha Signal Generation
    data['high_conviction_bullish'] = (
        data['positive_divergence'] & 
        data['strong_directional_volume'] & 
        data['efficient_vol_momentum'] &
        (data['momentum_5d'] > 0) & 
        (data['momentum_10d'] > 0)
    )
    
    data['high_conviction_bearish'] = (
        data['negative_divergence'] & 
        data['weak_directional_volume'] & 
        data['efficient_vol_momentum'] &
        (data['momentum_5d'] < 0) & 
        (data['momentum_10d'] < 0)
    )
    
    data['moderate_bullish'] = (
        data['strong_directional_volume'] & 
        (data['momentum_accel'] > 0) &
        (~data['volatility_regime_high'] & ~data['volatility_regime_low'])
    )
    
    data['moderate_bearish'] = (
        data['weak_directional_volume'] & 
        (data['momentum_accel'] < 0) &
        (~data['volatility_regime_high'] & ~data['volatility_regime_low'])
    )
    
    data['caution_signal'] = (
        data['inefficient_vol_momentum'] |
        ((data['momentum_5d'] > 0) & (data['momentum_10d'] < 0)) |
        ((data['momentum_5d'] < 0) & (data['momentum_10d'] > 0))
    )
    
    # Final alpha factor calculation
    alpha_signal = (
        data['high_conviction_bullish'].astype(int) * 2.0 +
        data['moderate_bullish'].astype(int) * 1.0 +
        data['high_conviction_bearish'].astype(int) * -2.0 +
        data['moderate_bearish'].astype(int) * -1.0 +
        data['caution_signal'].astype(int) * -0.5
    )
    
    # Add momentum strength weighting
    momentum_strength = (data['momentum_5d'] + data['momentum_10d']) / 2
    alpha_signal = alpha_signal * (1 + abs(momentum_strength))
    
    return alpha_signal
