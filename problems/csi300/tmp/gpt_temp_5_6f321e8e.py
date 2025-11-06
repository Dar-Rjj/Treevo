import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Volatility-Adjusted Momentum Divergence
    # Calculate Price Momentum (ROC 10 days)
    momentum = df['close'].pct_change(periods=10)
    
    # Calculate Volatility (Average True Range)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Compute Volatility-Adjusted Momentum (Momentum / ATR)
    vol_adj_momentum = momentum / atr
    
    # Calculate Volume Trend (Linear Regression Slope)
    def volume_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series.values)
        return slope
    
    volume_trend = df['volume'].rolling(window=10).apply(volume_slope, raw=False)
    
    # Compute Divergence (Vol-Adj Momentum - Volume Trend)
    divergence = vol_adj_momentum - volume_trend
    
    # Amount-Based Price Fractality
    # Calculate Fractal Dimension (Hurst Exponent Approximation)
    def hurst_approx(series):
        if len(series) < 20:
            return np.nan
        lags = range(2, 10)
        tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    fractal_dim = df['close'].rolling(window=30).apply(hurst_approx, raw=False)
    
    # Compute Amount Volatility (Rolling Std Dev)
    amount_vol = df['amount'].rolling(window=20).std()
    
    # Weight Fractal Dimension by Amount Volatility
    fractality_weighted = fractal_dim * amount_vol
    
    # Open-Close Imbalance Persistence
    # Calculate Daily Imbalance ((Close - Open) / Open)
    daily_imbalance = (df['close'] - df['open']) / df['open']
    
    # Track Sign Consistency Streak
    def sign_streak(series):
        if len(series) < 2:
            return 1
        current_sign = np.sign(series.iloc[-1])
        streak = 1
        for i in range(len(series)-2, -1, -1):
            if np.sign(series.iloc[i]) == current_sign:
                streak += 1
            else:
                break
        return streak
    
    sign_streak_length = daily_imbalance.rolling(window=10).apply(sign_streak, raw=False)
    
    # Compute Volume Ratio (Current / Recent Average)
    volume_ratio = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Multiply Streak Length by Volume Ratio
    imbalance_persistence = sign_streak_length * volume_ratio
    
    # Volume-Amount Dispersion Ratio
    # Calculate Volume Dispersion (Rolling Std Dev)
    volume_disp = df['volume'].rolling(window=15).std()
    
    # Calculate Amount Dispersion (Rolling Std Dev)
    amount_disp = df['amount'].rolling(window=15).std()
    
    # Compute Ratio (Volume Dispersion / Amount Dispersion)
    dispersion_ratio = volume_disp / amount_disp
    
    # High-Low Compression Breakout
    # Measure Price Compression (Range % of Close)
    price_range = (df['high'] - df['low']) / df['close']
    
    # Track Compression Duration (Low-Range Days)
    def compression_duration(series):
        threshold = series.rolling(window=10).mean().iloc[-1] * 0.7
        duration = 0
        for i in range(len(series)-1, -1, -1):
            if series.iloc[i] < threshold:
                duration += 1
            else:
                break
        return duration
    
    compression_days = price_range.rolling(window=15).apply(compression_duration, raw=False)
    
    # Detect Volume Surge (vs Compression Average)
    compression_vol_avg = df['volume'].rolling(window=10).mean()
    volume_surge = df['volume'] > compression_vol_avg * 1.5
    
    # Check Amount Consistency (> Compression Median)
    compression_amount_median = df['amount'].rolling(window=10).median()
    amount_consistent = df['amount'] > compression_amount_median
    
    # Combine breakout signals
    breakout_signal = compression_days * volume_surge.astype(int) * amount_consistent.astype(int)
    
    # Close-Based Mean Reversion Asymmetry
    # Calculate Deviation from Rolling Median
    price_median = df['close'].rolling(window=20).median()
    deviation = (df['close'] - price_median) / price_median
    
    # Separate Positive/Negative Deviations
    pos_deviation = deviation.where(deviation > 0, 0)
    neg_deviation = deviation.where(deviation < 0, 0).abs()
    
    # Compute Recovery Speed (Days to Return)
    def recovery_speed(series, is_positive=True):
        if len(series) < 5:
            return np.nan
        current_val = series.iloc[-1]
        if (is_positive and current_val <= 0) or (not is_positive and current_val >= 0):
            return 0
        
        days = 0
        for i in range(len(series)-2, -1, -1):
            days += 1
            if (is_positive and series.iloc[i] <= 0) or (not is_positive and series.iloc[i] >= 0):
                return days
        return np.nan
    
    upside_recovery = deviation.rolling(window=10).apply(
        lambda x: recovery_speed(x, True), raw=False)
    downside_recovery = deviation.rolling(window=10).apply(
        lambda x: recovery_speed(x, False), raw=False)
    
    # Compare Upside vs Downside Recovery
    mean_reversion_asymmetry = downside_recovery - upside_recovery
    
    # Volume-Weighted Price Acceleration
    # Calculate Multi-Timeframe Acceleration (ROC Diff)
    roc_5 = df['close'].pct_change(periods=5)
    roc_10 = df['close'].pct_change(periods=10)
    acceleration = roc_5 - roc_10
    
    # Compute Volume Percentile (Within Lookback)
    def volume_percentile(series):
        if len(series) < 2:
            return 0.5
        current_vol = series.iloc[-1]
        return np.sum(series < current_vol) / len(series)
    
    vol_percentile = df['volume'].rolling(window=20).apply(volume_percentile, raw=False)
    
    # Multiply Acceleration by Volume Percentile
    vol_weighted_acceleration = acceleration * vol_percentile
    
    # Combine all factors with equal weights
    combined_factor = (
        divergence.fillna(0) +
        fractality_weighted.fillna(0) +
        imbalance_persistence.fillna(0) +
        dispersion_ratio.fillna(0) +
        breakout_signal.fillna(0) +
        mean_reversion_asymmetry.fillna(0) +
        vol_weighted_acceleration.fillna(0)
    )
    
    return combined_factor
