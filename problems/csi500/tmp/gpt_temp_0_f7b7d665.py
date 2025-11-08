import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Divergence factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Asymmetric Momentum Component
    # Short-Term Momentum (3-day)
    mom_3d = data['close'] / data['close'].shift(3) - 1
    daily_range_vol = (data['high'] - data['low']) / data['close'].shift(1)
    vol_3d_std = daily_range_vol.rolling(window=5, min_periods=3).std()
    mom_3d_scaled = mom_3d / (vol_3d_std + 1e-8)
    
    # Medium-Term Momentum (8-day)
    mom_8d = data['close'] / data['close'].shift(8) - 1
    true_range = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hc': abs(data['high'] - data['close'].shift(1)),
        'lc': abs(data['low'] - data['close'].shift(1))
    }).max(axis=1) / data['close'].shift(1)
    vol_8d_std = true_range.rolling(window=8, min_periods=5).std()
    mom_8d_scaled = mom_8d / (vol_8d_std + 1e-8)
    
    # Long-Term Momentum (20-day)
    mom_20d = data['close'] / data['close'].shift(20) - 1
    gk_vol = np.sqrt(
        0.5 * (np.log(data['high'] / data['low']))**2 - 
        (2*np.log(2)-1) * (np.log(data['close'] / data['open']))**2
    )
    vol_20d_std = gk_vol.rolling(window=10, min_periods=7).std()
    mom_20d_scaled = mom_20d / (vol_20d_std + 1e-8)
    
    # Volume Regime Detection
    # Multi-Timeframe Volume Analysis
    vol_5d_mean = data['volume'].rolling(window=5, min_periods=3).mean()
    vol_5d_std = data['volume'].rolling(window=5, min_periods=3).std()
    vol_zscore = (data['volume'] - vol_5d_mean) / (vol_5d_std + 1e-8)
    
    vol_15d_percentile = data['volume'].rolling(window=15, min_periods=10).apply(
        lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5
    )
    
    vol_30d_mean = data['volume'].rolling(window=30, min_periods=20).mean()
    vol_trend = data['volume'] / vol_30d_mean - 1
    
    # Volume Regime Classification
    high_vol_regime = (vol_zscore > 1) & (vol_15d_percentile > 0.7)
    low_vol_regime = (vol_zscore < -0.5) & (vol_15d_percentile < 0.3)
    normal_vol_regime = ~(high_vol_regime | low_vol_regime)
    
    # Nonlinear Transformation Layer
    # Momentum Combination
    mom_weights = [0.4, 0.35, 0.25]
    mom_signs = np.sign(mom_3d_scaled + mom_8d_scaled + mom_20d_scaled)
    
    mom_combined = (
        np.abs(mom_3d_scaled)**mom_weights[0] * 
        np.abs(mom_8d_scaled)**mom_weights[1] * 
        np.abs(mom_20d_scaled)**mom_weights[2]
    ) ** (1 / sum(mom_weights))
    mom_combined = mom_combined * mom_signs
    
    # Volume Signal Transformation
    vol_signal = pd.Series(index=data.index, dtype=float)
    vol_signal[high_vol_regime] = np.tanh(vol_trend[high_vol_regime] * 2)
    vol_signal[low_vol_regime] = np.arctan(vol_15d_percentile[low_vol_regime] * 2)
    vol_signal[normal_vol_regime] = np.clip(vol_zscore[normal_vol_regime], -2, 2)
    
    # Bounded Output
    base_signal = mom_combined * vol_signal
    bounded_signal = 1.6 * (1 / (1 + np.exp(-base_signal)) - 0.5)
    
    # Divergence Enhancement
    # Price-Volume Divergence
    price_5d_ret = data['close'].pct_change(5)
    vol_5d_ma = data['volume'].rolling(window=5).mean()
    pv_corr = price_5d_ret.rolling(window=10).corr(vol_5d_ma)
    
    divergence_multiplier = np.where(pv_corr < 0, 1.5, 1.0)
    
    # Multi-Timeframe Alignment
    mom_signs_3_8 = (np.sign(mom_3d_scaled) == np.sign(mom_8d_scaled))
    mom_signs_8_20 = (np.sign(mom_8d_scaled) == np.sign(mom_20d_scaled))
    
    alignment_multiplier = np.where(mom_signs_3_8 & mom_signs_8_20, 1.2, 
                                   np.where(~mom_signs_3_8 & ~mom_signs_8_20, 0.8, 1.0))
    
    # Final factor calculation
    final_factor = bounded_signal * divergence_multiplier * alignment_multiplier
    
    return final_factor
