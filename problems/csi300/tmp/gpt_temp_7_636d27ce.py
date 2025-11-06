import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Regime-Adaptive Multi-Timeframe Alpha Factor
    Combines volatility regime detection with price-volume asymmetry and trend persistence
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility Regime Detection
    # Parkinson Volatility (daily)
    parkinson_vol = (np.log(data['high'] / data['low']) ** 2) / (4 * np.log(2))
    
    # Short-term volatility (5-day average of Parkinson)
    short_term_vol = parkinson_vol.rolling(window=5, min_periods=3).mean()
    
    # Medium-term volatility (20-day std of returns)
    returns = data['close'].pct_change()
    medium_term_vol = returns.rolling(window=20, min_periods=15).std()
    
    # Regime classification
    vol_diff = short_term_vol - medium_term_vol
    threshold = medium_term_vol.rolling(window=20, min_periods=15).std() * 0.5
    
    regime = pd.Series(index=data.index, dtype='object')
    regime[vol_diff > threshold] = 'high'
    regime[vol_diff < -threshold] = 'low'
    regime[regime.isna()] = 'transition'
    
    # 2. Price-Volume Asymmetry Analysis
    # Multi-Timeframe Volume Pressure
    # 3-day Upside Volume Ratio
    up_days_3d = data['close'] > data['close'].shift(1)
    up_volume_3d = data['volume'].where(up_days_3d)
    up_volume_avg_3d = up_volume_3d.rolling(window=3, min_periods=2).mean()
    upside_volume_ratio = data['volume'] / up_volume_avg_3d
    
    # 3-day Downside Volume Ratio
    down_days_3d = data['close'] < data['close'].shift(1)
    down_volume_3d = data['volume'].where(down_days_3d)
    down_volume_avg_3d = down_volume_3d.rolling(window=3, min_periods=2).mean()
    downside_volume_ratio = data['volume'] / down_volume_avg_3d
    
    # 10-day Volume Asymmetry
    up_days_10d = data['close'] > data['close'].shift(1)
    down_days_10d = data['close'] < data['close'].shift(1)
    
    up_volume_cum_10d = data['volume'].where(up_days_10d).rolling(window=10, min_periods=8).sum()
    down_volume_cum_10d = data['volume'].where(down_days_10d).rolling(window=10, min_periods=8).sum()
    volume_asymmetry = up_volume_cum_10d / (down_volume_cum_10d + 1e-8)
    
    # Price-Volume Divergence
    def linear_trend(series, window):
        """Calculate linear regression slope for given window"""
        def calc_slope(x):
            if len(x) < 2:
                return np.nan
            return stats.linregress(np.arange(len(x)), x)[0]
        return series.rolling(window=window, min_periods=int(window*0.7)).apply(calc_slope, raw=False)
    
    price_trend_3d = linear_trend(data['close'], 3)
    volume_trend_3d = linear_trend(data['volume'], 3)
    price_trend_10d = linear_trend(data['close'], 10)
    volume_trend_10d = linear_trend(data['volume'], 10)
    
    divergence_3d = price_trend_3d - volume_trend_3d
    divergence_10d = price_trend_10d - volume_trend_10d
    divergence_score = (divergence_3d + divergence_10d) / 2
    
    # 3. Trend Persistence Measurement
    # Multi-Scale Trend Strength
    trend_3d = linear_trend(data['close'], 3)
    trend_10d = linear_trend(data['close'], 10)
    trend_20d = linear_trend(data['close'], 20)
    
    # Trend Acceleration
    accel_3_10 = trend_3d - trend_10d
    accel_10_20 = trend_10d - trend_20d
    acceleration_score = accel_3_10 + accel_10_20
    
    # Volume-Confirmed Trend
    trend_direction_10d = np.sign(trend_10d)
    confirming_days = (np.sign(data['close'].diff()) == trend_direction_10d) & (trend_direction_10d != 0)
    contradicting_days = (np.sign(data['close'].diff()) == -trend_direction_10d) & (trend_direction_10d != 0)
    
    confirming_volume = data['volume'].where(confirming_days).rolling(window=10, min_periods=8).sum()
    contradicting_volume = data['volume'].where(contradicting_days).rolling(window=10, min_periods=8).sum()
    confirmation_ratio = confirming_volume / (contradicting_volume + 1e-8)
    
    # 4. Regime-Adaptive Signal Combination
    # Normalize components
    volume_asymmetry_norm = (volume_asymmetry - volume_asymmetry.rolling(50).mean()) / (volume_asymmetry.rolling(50).std() + 1e-8)
    divergence_score_norm = (divergence_score - divergence_score.rolling(50).mean()) / (divergence_score.rolling(50).std() + 1e-8)
    trend_strength_norm = (trend_10d - trend_10d.rolling(50).mean()) / (trend_10d.rolling(50).std() + 1e-8)
    acceleration_norm = (acceleration_score - acceleration_score.rolling(50).mean()) / (acceleration_score.rolling(50).std() + 1e-8)
    confirmation_norm = (confirmation_ratio - confirmation_ratio.rolling(50).mean()) / (confirmation_ratio.rolling(50).std() + 1e-8)
    
    # Combined trend persistence (weighted average)
    trend_persistence = 0.5 * trend_strength_norm + 0.3 * acceleration_norm + 0.2 * confirmation_norm
    
    # Regime-specific weighting
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    # High Volatility Regime
    high_mask = regime == 'high'
    alpha_signal[high_mask] = (
        0.6 * volume_asymmetry_norm[high_mask] +
        0.3 * trend_persistence[high_mask] +
        0.1 * divergence_score_norm[high_mask]
    )
    
    # Low Volatility Regime
    low_mask = regime == 'low'
    alpha_signal[low_mask] = (
        0.6 * trend_persistence[low_mask] +
        0.3 * divergence_score_norm[low_mask] +
        0.1 * volume_asymmetry_norm[low_mask]
    )
    
    # Transition Regime
    trans_mask = regime == 'transition'
    alpha_signal[trans_mask] = (
        0.33 * volume_asymmetry_norm[trans_mask] +
        0.33 * trend_persistence[trans_mask] +
        0.33 * divergence_score_norm[trans_mask]
    ) * 0.7  # Reduced signal strength
    
    # Final normalization
    alpha_signal = (alpha_signal - alpha_signal.rolling(50).mean()) / (alpha_signal.rolling(50).std() + 1e-8)
    
    return alpha_signal
