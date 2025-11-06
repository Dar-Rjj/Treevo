import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Volatility-Adjusted Momentum Divergence
    # Calculate Price Momentum (10-day ROC)
    momentum_10d = df['close'].pct_change(periods=10)
    
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10d = true_range.rolling(window=10).mean()
    
    # Volatility-Adjusted Momentum
    vol_adj_momentum = momentum_10d / atr_10d
    
    # Calculate Volume Pattern using Linear Regression Slope (10-day window)
    def volume_slope(volume_series):
        if len(volume_series) < 2:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = linregress(x, volume_series)
        return slope
    
    volume_trend = df['volume'].rolling(window=10).apply(volume_slope, raw=False)
    
    # Volatility-Adjusted Momentum Divergence
    momentum_divergence = vol_adj_momentum - volume_trend
    
    # Amount-Based Price Fractality
    # Calculate High-Low Range
    hl_range = (df['high'] - df['low']) / df['close']
    
    # Compute Hurst Exponent Approximation using Rescaled Range (20-day window)
    def hurst_approx(price_series):
        if len(price_series) < 10:
            return np.nan
        lags = range(2, min(10, len(price_series)))
        tau = []
        for lag in lags:
            rs = (price_series.diff(lag).std() / price_series.std()) if price_series.std() > 0 else 0
            tau.append(rs)
        if len(tau) > 1:
            x = np.log(lags[:len(tau)])
            y = np.log(tau)
            slope, _, _, _, _ = linregress(x, y)
            return slope
        return np.nan
    
    fractal_dim = df['close'].rolling(window=20).apply(hurst_approx, raw=False)
    
    # Weight by Amount Volatility
    amount_vol = df['amount'].rolling(window=10).std()
    fractality_weighted = fractal_dim * amount_vol
    
    # Open-Close Imbalance Persistence
    # Calculate Daily Imbalance
    daily_imbalance = (df['close'] - df['open']) / df['open']
    
    # Compute Sign Consistency (Maximum Streak Length over 20 days)
    def max_streak_length(imbalance_series):
        if len(imbalance_series) < 2:
            return 0
        signs = np.sign(imbalance_series)
        current_streak = 1
        max_streak = 1
        
        for i in range(1, len(signs)):
            if signs[i] == signs[i-1] and signs[i] != 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        return max_streak
    
    streak_length = daily_imbalance.rolling(window=20).apply(max_streak_length, raw=False)
    
    # Weight by Volume Confirmation
    volume_ratio = df['volume'] / df['volume'].rolling(window=10).mean()
    imbalance_persistence = streak_length * volume_ratio
    
    # Volume-Amount Dispersion Ratio
    volume_dispersion = df['volume'].rolling(window=10).std()
    amount_dispersion = df['amount'].rolling(window=10).std()
    dispersion_ratio = volume_dispersion / (amount_dispersion + 1e-8)
    
    # High-Low Compression Breakout
    # Measure Price Compression
    price_range_pct = (df['high'] - df['low']) / df['close']
    
    # Track Compression Duration (consecutive days below 20th percentile of 20-day range)
    range_20pct = price_range_pct.rolling(window=20).quantile(0.2)
    low_range_mask = price_range_pct < range_20pct
    
    def compression_duration(mask_series):
        if len(mask_series) < 2:
            return 0
        current_streak = 0
        for i in range(len(mask_series)-1, -1, -1):
            if mask_series.iloc[i]:
                current_streak += 1
            else:
                break
        return current_streak
    
    compression_days = low_range_mask.rolling(window=10).apply(compression_duration, raw=False)
    
    # Breakout Confirmation
    compression_volume_avg = df['volume'].rolling(window=10).mean().shift(1)
    compression_amount_median = df['amount'].rolling(window=10).median().shift(1)
    
    volume_surge = df['volume'] > compression_volume_avg
    amount_consistent = df['amount'] > compression_amount_median
    
    breakout_quality = compression_days * volume_surge.astype(int) * amount_consistent.astype(int)
    
    # Close-Based Mean Reversion Asymmetry
    # Calculate Short-Term Deviation from 10-day median
    median_10d = df['close'].rolling(window=10).median()
    deviation = (df['close'] - median_10d) / median_10d
    
    # Separate positive and negative deviations
    pos_deviation = deviation.where(deviation > 0, 0)
    neg_deviation = deviation.where(deviation < 0, 0)
    
    # Compute recovery speed (simplified as inverse of absolute deviation)
    upside_recovery = 1 / (abs(pos_deviation) + 1e-8)
    downside_recovery = 1 / (abs(neg_deviation) + 1e-8)
    
    mean_reversion_asymmetry = upside_recovery - downside_recovery
    
    # Volume-Weighted Price Acceleration Profile
    # Calculate Multi-Timeframe Acceleration
    roc_3d = df['close'].pct_change(periods=3)
    roc_10d = df['close'].pct_change(periods=10)
    acceleration = roc_3d - roc_10d
    
    # Apply Volume-Based Weighting
    volume_percentile = df['volume'].rolling(window=20).rank(pct=True)
    volume_weighted_acceleration = acceleration * volume_percentile
    
    # Combine all factors with equal weights
    final_factor = (
        momentum_divergence.fillna(0) +
        fractality_weighted.fillna(0) +
        imbalance_persistence.fillna(0) +
        dispersion_ratio.fillna(0) +
        breakout_quality.fillna(0) +
        mean_reversion_asymmetry.fillna(0) +
        volume_weighted_acceleration.fillna(0)
    ) / 7
    
    return final_factor
