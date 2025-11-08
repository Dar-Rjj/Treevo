import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Convergence-Divergence with Regime Switching
    
    # Price Trend Acceleration
    # Compute 5-day Price Slope (t-4 to t)
    price_slope_5 = (df['close'] - df['close'].shift(4)) / df['close'].shift(4)
    
    # Compute 10-day Price Slope (t-9 to t)
    price_slope_10 = (df['close'] - df['close'].shift(9)) / df['close'].shift(9)
    
    # Calculate Slope Ratio (5-day / 10-day)
    price_slope_ratio = price_slope_5 / price_slope_10.replace(0, np.nan)
    
    # Volume Trend Acceleration
    # Compute 5-day Volume Slope (t-4 to t)
    volume_slope_5 = (df['volume'] - df['volume'].shift(4)) / df['volume'].shift(4).replace(0, np.nan)
    
    # Compute 10-day Volume Slope (t-9 to t)
    volume_slope_10 = (df['volume'] - df['volume'].shift(9)) / df['volume'].shift(9).replace(0, np.nan)
    
    # Calculate Volume Slope Ratio (5-day / 10-day)
    volume_slope_ratio = volume_slope_5 / volume_slope_10.replace(0, np.nan)
    
    # Price-Volume Convergence
    price_volume_convergence = price_slope_ratio * volume_slope_ratio
    
    # Detect Market Regimes
    # Identify Volatility Regime
    # Calculate Daily Range % ((High-Low)/Close at t)
    daily_range_pct = (df['high'] - df['low']) / df['close']
    
    # Compute 20-day Median Range (t-19 to t)
    median_range_20 = daily_range_pct.rolling(window=20, min_periods=10).median()
    
    # Create Regime Indicator (1 if Range > Median, else 0)
    regime_indicator = (daily_range_pct > median_range_20).astype(int)
    
    # Calculate Volume-Price Divergence
    # Compute 10-day Volume-Price Correlation (t-9 to t)
    def rolling_corr(x, y, window):
        return x.rolling(window=window).corr(y)
    
    volume_price_corr = rolling_corr(df['volume'], df['close'], window=10)
    
    # Multiply by Price Trend Sign
    price_trend_sign = np.sign(price_slope_10)
    divergence_with_sign = volume_price_corr * price_trend_sign
    
    # Multiply by Price Trend Strength
    price_trend_strength = abs(price_slope_10)
    divergence_strength = divergence_with_sign * price_trend_strength
    
    # Generate Adaptive Alpha Factor
    # Create Base Factor (Price-Volume Convergence × Regime Indicator)
    base_factor = price_volume_convergence * regime_indicator
    
    # Apply Divergence Weighting (Base Factor × Divergence Strength)
    weighted_factor = base_factor * divergence_strength
    
    # Apply Adaptive Smoothing
    # Select Window Size (3-day if High Volatility, 5-day otherwise)
    window_sizes = np.where(regime_indicator == 1, 3, 5)
    
    # Compute Moving Average with Selected Window
    def adaptive_rolling_mean(series, windows):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i >= max(windows):
                window = windows[i]
                result.iloc[i] = series.iloc[i-window+1:i+1].mean()
        return result
    
    alpha_factor = adaptive_rolling_mean(weighted_factor, window_sizes)
    
    return alpha_factor
