import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a comprehensive alpha factor combining multiple market insights
    """
    # Price-Volume Divergence Factor
    # Calculate price momentum
    price_mom_3d = df['close'] - df['close'].shift(3)
    price_mom_10d = df['close'] - df['close'].shift(10)
    price_mom_20d = df['close'] - df['close'].shift(20)
    
    # Calculate volume momentum
    volume_mom_3d = df['volume'] - df['volume'].shift(3)
    volume_mom_10d = df['volume'] - df['volume'].shift(10)
    volume_mom_20d = df['volume'] - df['volume'].shift(20)
    
    # Compute divergence signals
    div_3d = price_mom_3d * volume_mom_3d
    div_10d = price_mom_10d * volume_mom_10d
    div_20d = price_mom_20d * volume_mom_20d
    
    # Weighted divergence factor
    price_volume_divergence = 0.5 * div_3d + 0.3 * div_10d + 0.2 * div_20d
    
    # Volatility-Adjusted Range Momentum
    daily_range = df['high'] - df['low']
    
    # Range momentum
    range_mom_5d = daily_range - daily_range.shift(5)
    range_mom_10d = daily_range - daily_range.shift(10)
    
    # Volatility proxy
    avg_range_10d = daily_range.rolling(window=10).mean()
    
    # Volatility-adjusted range momentum
    vol_adj_range_mom = (range_mom_5d + range_mom_10d) / avg_range_10d
    
    # Volume-Weighted Price Efficiency
    # Price efficiency
    daily_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    avg_efficiency_5d = daily_efficiency.rolling(window=5).mean()
    efficiency_mom = daily_efficiency - daily_efficiency.shift(5)
    
    # Volume weighting
    volume_ratio_5d = df['volume'] / (df['volume'].rolling(window=5).mean())
    volume_trend = df['volume'] - df['volume'].shift(5)
    
    # Volume-weighted efficiency
    volume_weighted_efficiency = efficiency_mom * volume_ratio_5d * volume_trend
    
    # Multi-Timeframe Breakout Confirmation
    # Breakout signals
    breakout_5d = (df['close'] > df['close'].shift(1).rolling(window=4).max()).astype(int)
    breakout_10d = (df['close'] > df['close'].shift(1).rolling(window=9).max()).astype(int)
    breakout_20d = (df['close'] > df['close'].shift(1).rolling(window=19).max()).astype(int)
    
    # Volume confirmation
    volume_surge_5d = (df['volume'] > 1.5 * df['volume'].shift(1).rolling(window=4).mean()).astype(int)
    volume_surge_10d = (df['volume'] > 1.5 * df['volume'].shift(1).rolling(window=9).mean()).astype(int)
    volume_surge_20d = (df['volume'] > 1.5 * df['volume'].shift(1).rolling(window=19).mean()).astype(int)
    
    # Breakout score
    breakout_score = (breakout_5d * volume_surge_5d + 
                     breakout_10d * volume_surge_10d + 
                     breakout_20d * volume_surge_20d)
    
    # Regime-Adaptive Momentum Quality
    # Identify volatility regime
    range_percentile = avg_range_10d.rolling(window=50).apply(
        lambda x: (x.iloc[-1] > np.percentile(x[:-1], 70)) * 2 + 
                 (x.iloc[-1] < np.percentile(x[:-1], 30)) * 1, 
        raw=False
    )
    
    # Calculate regime-specific momentum
    regime_momentum = pd.Series(index=df.index, dtype=float)
    high_vol_mask = range_percentile == 2
    low_vol_mask = range_percentile == 1
    normal_vol_mask = range_percentile == 0
    
    regime_momentum[high_vol_mask] = (df['close'] - df['close'].shift(5))[high_vol_mask] / avg_range_10d[high_vol_mask]
    regime_momentum[low_vol_mask] = (df['close'] - df['close'].shift(10))[low_vol_mask]
    regime_momentum[normal_vol_mask] = (df['close'] - df['close'].shift(7))[normal_vol_mask] / np.sqrt(avg_range_10d[normal_vol_mask])
    
    # Compute momentum consistency
    price_change_sign = np.sign(df['close'].diff())
    sign_consistency_5d = price_change_sign.rolling(window=5).apply(lambda x: len(set(x.dropna())) if len(x.dropna()) > 0 else 1, raw=False)
    sign_consistency_10d = price_change_sign.rolling(window=10).apply(lambda x: len(set(x.dropna())) if len(x.dropna()) > 0 else 1, raw=False)
    
    # Adaptive momentum quality
    adaptive_momentum_quality = regime_momentum * (sign_consistency_5d + sign_consistency_10d)
    
    # Combine all factors with equal weighting
    combined_factor = (
        price_volume_divergence.fillna(0) +
        vol_adj_range_mom.fillna(0) +
        volume_weighted_efficiency.fillna(0) +
        breakout_score.fillna(0) +
        adaptive_momentum_quality.fillna(0)
    )
    
    return combined_factor
