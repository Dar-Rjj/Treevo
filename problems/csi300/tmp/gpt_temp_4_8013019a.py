import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum with Volume Acceleration
    # Compute Short-Term Momentum
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    momentum_10d = df['close'] / df['close'].shift(10) - 1
    
    # Compute Volume Acceleration
    vol_accel_5d = df['volume'] / df['volume'].shift(5) - 1
    vol_accel_10d = df['volume'] / df['volume'].shift(10) - 1
    
    # Combine Signals
    mom_vol_5d = momentum_5d * vol_accel_5d
    mom_vol_10d = momentum_10d * vol_accel_10d
    
    # Volatility-Adjusted Range Breakout
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Compute Volatility Normalization
    atr_20d = true_range.rolling(window=20).mean()
    range_ratio = (df['high'] - df['low']) / atr_20d
    
    # Detect Breakout Patterns
    high_breakout = (df['close'] > df['open']).astype(float) * range_ratio
    low_breakdown = (df['close'] < df['open']).astype(float) * range_ratio
    breakout_signal = high_breakout - low_breakdown
    
    # Liquidity-Weighted Price Reversal
    # Identify Recent Extremes
    high_10d = df['high'].rolling(window=10).max()
    low_10d = df['low'].rolling(window=10).min()
    
    # Calculate Reversal Strength
    dist_from_high = high_10d - df['close']
    dist_from_low = df['close'] - low_10d
    
    # Apply Liquidity Weighting
    avg_vol_5d = df['volume'].rolling(window=5).mean()
    high_reversal = dist_from_high * df['volume'] / (abs(dist_from_high) + 1e-8)
    low_reversal = dist_from_low * df['volume'] / (abs(dist_from_low) + 1e-8)
    reversal_signal = high_reversal - low_reversal
    
    # Intraday Strength Persistence
    # Calculate Intraday Strength
    intraday_return = (df['close'] - df['open']) / df['open']
    vol_20d = df['close'].rolling(window=20).std()
    adj_intraday_return = intraday_return / (vol_20d + 1e-8)
    
    # Measure Persistence
    sign_series = np.sign(adj_intraday_return)
    consecutive_count = sign_series.groupby((sign_series != sign_series.shift(1)).cumsum()).cumcount() + 1
    persistence_strength = consecutive_count * adj_intraday_return
    
    # Volume Confirmation
    avg_vol_20d = df['volume'].rolling(window=20).mean()
    vol_ratio = df['volume'] / (avg_vol_20d + 1e-8)
    intraday_persistence = persistence_strength * vol_ratio
    
    # Price-Volume Divergence Oscillator
    # Compute Price Trend
    def calc_slope(series, window):
        x = np.arange(window)
        slopes = series.rolling(window=window).apply(
            lambda y: np.polyfit(x, y, 1)[0] if not y.isna().any() else np.nan, 
            raw=False
        )
        return slopes
    
    price_slope_5d = calc_slope(df['close'], 5)
    price_slope_10d = calc_slope(df['close'], 10)
    
    # Compute Volume Trend
    volume_slope_5d = calc_slope(df['volume'], 5)
    volume_slope_10d = calc_slope(df['volume'], 10)
    
    # Detect Divergence
    # Normalize slopes by their rolling standard deviation
    price_slope_5d_norm = price_slope_5d / (price_slope_5d.rolling(window=20).std() + 1e-8)
    volume_slope_5d_norm = volume_slope_5d / (volume_slope_5d.rolling(window=20).std() + 1e-8)
    price_slope_10d_norm = price_slope_10d / (price_slope_10d.rolling(window=20).std() + 1e-8)
    volume_slope_10d_norm = volume_slope_10d / (volume_slope_10d.rolling(window=20).std() + 1e-8)
    
    divergence_5d = price_slope_5d_norm - volume_slope_5d_norm
    divergence_10d = price_slope_10d_norm - volume_slope_10d_norm
    
    # Combine all factors with equal weights
    factor = (mom_vol_5d.fillna(0) + mom_vol_10d.fillna(0) + 
              breakout_signal.fillna(0) + reversal_signal.fillna(0) + 
              intraday_persistence.fillna(0) + divergence_5d.fillna(0) + 
              divergence_10d.fillna(0))
    
    return factor
