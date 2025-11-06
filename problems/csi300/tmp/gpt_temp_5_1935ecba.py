import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Adjusted Trend Strength
    # Calculate Short-Term Trend (5-day Linear Regression Slope)
    def linear_regression_slope(series, window):
        slopes = np.full(len(series), np.nan)
        for i in range(window-1, len(series)):
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(window)
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes[i] = slope
        return slopes
    
    trend_5d = linear_regression_slope(data['close'], 5)
    
    # Calculate Medium-Term Volatility (20-day Average True Range)
    def average_true_range(high, low, close, window):
        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            )
        )
        atr = tr.rolling(window=window, min_periods=1).mean()
        return atr
    
    atr_20d = average_true_range(data['high'], data['low'], data['close'], 20)
    
    # Adjust Trend by Volatility and Scale by Volume Activity
    vol_adj_trend = trend_5d / (atr_20d + 1e-8)
    volume_ratio = data['volume'].rolling(window=5, min_periods=1).mean() / \
                   data['volume'].rolling(window=20, min_periods=1).mean()
    factor1 = vol_adj_trend * volume_ratio
    
    # Price-Momentum Divergence Factor
    # Compute Price Momentum (10-day Rate of Change)
    price_roc = data['close'].pct_change(periods=10)
    
    # Compute Volume Momentum (10-day Volume Rate of Change)
    volume_roc = data['volume'].pct_change(periods=10)
    
    # Detect Divergence with Correlation Strength
    price_changes = data['close'].pct_change()
    volume_changes = data['volume'].pct_change()
    
    # Calculate 10-day rolling correlation
    correlation = price_changes.rolling(window=10, min_periods=1).corr(volume_changes)
    
    # Invert for negative correlation cases and multiply by direction agreement
    direction_agreement = np.sign(price_roc * volume_roc)
    correlation_strength = -correlation * direction_agreement
    factor2 = correlation_strength * np.abs(price_roc * volume_roc)
    
    # Intraday Pressure Accumulation
    # Calculate Buying Pressure
    close_above_open = (data['close'] > data['open']).astype(float)
    high_above_prev_close = (data['high'] > data['close'].shift(1)).astype(float)
    buying_pressure = (close_above_open + high_above_prev_close) * data['volume']
    
    # Calculate Selling Pressure
    close_below_open = (data['close'] < data['open']).astype(float)
    low_below_prev_close = (data['low'] < data['close'].shift(1)).astype(float)
    selling_pressure = (close_below_open + low_below_prev_close) * data['volume']
    
    # Compute Net Pressure with Exponential Decay
    net_pressure = buying_pressure - selling_pressure
    
    # Apply exponential decay over 5-day window
    decay_weights = np.exp(-np.arange(5) / 2.5)  # Half-life of ~2.5 days
    decay_weights = decay_weights / decay_weights.sum()
    
    pressure_accumulated = np.full(len(data), np.nan)
    for i in range(4, len(data)):
        if i >= 4:
            window_data = net_pressure.iloc[i-4:i+1].values
            pressure_accumulated[i] = np.sum(window_data * decay_weights)
    
    # Scale by Average Daily Range
    daily_range = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    factor3 = pressure_accumulated / (daily_range + 1e-8)
    
    # Liquidity-Adjusted Reversal Signal
    # Identify Overbought/Oversold Conditions
    rolling_high = data['close'].rolling(window=20, min_periods=1).max()
    rolling_low = data['close'].rolling(window=20, min_periods=1).min()
    price_position = (data['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
    
    # Calculate Percentile Rank
    def percentile_rank(series, window):
        ranks = np.full(len(series), np.nan)
        for i in range(window-1, len(series)):
            window_data = series.iloc[i-window+1:i+1]
            current_val = series.iloc[i]
            ranks[i] = (window_data <= current_val).sum() / len(window_data)
        return ranks
    
    overbought_oversold = percentile_rank(price_position, 20) - 0.5
    
    # Measure Liquidity Conditions (Volume Z-score)
    volume_mean = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_std = data['volume'].rolling(window=20, min_periods=1).std()
    volume_zscore = (data['volume'] - volume_mean) / (volume_std + 1e-8)
    
    # Generate Reversal Signal with Non-linear Transformation
    reversal_raw = overbought_oversold * volume_zscore
    factor4 = np.tanh(reversal_raw) * atr_20d
    
    # Multi-Timeframe Acceleration Factor
    # Compute Acceleration at Different Horizons
    mom_3d = data['close'].pct_change(periods=3)
    mom_5d = data['close'].pct_change(periods=5)
    mom_10d = data['close'].pct_change(periods=10)
    
    accel_3d = mom_3d - mom_3d.shift(3)
    accel_5d = mom_5d - mom_5d.shift(5)
    accel_10d = mom_10d - mom_10d.shift(10)
    
    # Weight by Consistency (Sign Concordance)
    sign_concordance = (np.sign(accel_3d) == np.sign(accel_5d)) & \
                       (np.sign(accel_5d) == np.sign(accel_10d))
    sign_concordance = sign_concordance.astype(float)
    
    # Combine accelerations with consistency weighting
    combined_accel = (accel_3d + accel_5d + accel_10d) / 3
    factor5 = combined_accel * sign_concordance * atr_20d
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'factor1': factor1,
        'factor2': factor2,
        'factor3': factor3,
        'factor4': factor4,
        'factor5': factor5
    })
    
    # Z-score normalization for each factor
    for col in factors.columns:
        factors[col] = (factors[col] - factors[col].mean()) / (factors[col].std() + 1e-8)
    
    # Equal-weighted combination
    final_factor = factors.mean(axis=1)
    
    return pd.Series(final_factor, index=data.index)
