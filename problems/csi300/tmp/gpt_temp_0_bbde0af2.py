import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Acceleration
    price_3d = data['close'] / data['close'].shift(3) - 1
    price_6d = data['close'] / data['close'].shift(6) - 1
    price_accel = price_3d - price_6d
    
    # Volume Acceleration
    vol_3d = data['volume'] / data['volume'].shift(3) - 1
    vol_6d = data['volume'] / data['volume'].shift(6) - 1
    vol_accel = vol_3d - vol_6d
    
    # Regime Detection - ATR Calculation
    tr = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hc': abs(data['high'] - data['close'].shift(1)),
        'lc': abs(data['low'] - data['close'].shift(1))
    }).max(axis=1)
    atr_10d = tr.rolling(window=10).mean()
    avg_vol_10d = data['volume'].rolling(window=10).mean()
    
    # Regime Classification
    atr_percentile = atr_10d.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    vol_percentile = avg_vol_10d.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    high_vol_regime = (atr_percentile > 0.7).astype(int)
    high_volume_regime = (vol_percentile > 0.7).astype(int)
    normal_regime = ((atr_percentile <= 0.7) & (vol_percentile <= 0.7)).astype(int)
    
    # Regime-Specific Weights
    price_weight = high_vol_regime * 0.7 + high_volume_regime * 0.3 + normal_regime * 0.5
    volume_weight = high_vol_regime * 0.3 + high_volume_regime * 0.7 + normal_regime * 0.5
    
    # Momentum Quality Assessment
    # Intraday Efficiency Ratio
    intraday_efficiency = abs((data['close'] - data['open']) / (data['high'] - data['low']))
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Gap Persistence
    overnight_gap = data['open'] / data['close'].shift(1) - 1
    gap_persistence = overnight_gap.rolling(window=3).apply(lambda x: len([i for i in x if abs(i) > 0.005]) / 3)
    
    # Combined Quality Score
    quality_score = intraday_efficiency * gap_persistence
    
    # Factor Integration
    accel_product = price_accel * vol_accel
    regime_weighted = price_weight * price_accel + volume_weight * vol_accel
    
    # Scale by recent price range
    price_range_5d = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / data['close']
    scaled_factor = regime_weighted * quality_score / price_range_5d
    
    # Signal Enhancement
    # Trend Consistency Filter
    price_3d_sign = np.sign(price_3d)
    price_6d_sign = np.sign(price_6d)
    trend_consistent = (price_3d_sign == price_6d_sign).astype(int)
    trend_multiplier = 1 + (trend_consistent * 0.3)
    
    # Volume Confirmation
    vol_confirmation = (np.sign(price_accel) == np.sign(vol_accel)).astype(int)
    confirmation_bonus = 1 + (vol_confirmation * 0.2)
    
    # Final Factor
    final_factor = scaled_factor * trend_multiplier * confirmation_bonus
    
    return final_factor
