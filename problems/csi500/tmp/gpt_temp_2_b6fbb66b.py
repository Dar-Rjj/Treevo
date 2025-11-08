import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Based Momentum & Reversion
    # Short-Term Return Acceleration
    ret_1d = data['close'].pct_change(1)
    ret_3d = data['close'].pct_change(3)
    acceleration = (ret_3d - ret_1d) / (abs(ret_1d) + 1e-8)
    
    # Volatility-Adjusted Momentum
    high_low_range = data['high'] - data['low']
    high_prev_close = abs(data['high'] - data['close'].shift(1))
    low_prev_close = abs(data['low'] - data['close'].shift(1))
    true_range = np.maximum(high_low_range, np.maximum(high_prev_close, low_prev_close))
    atr_20d = true_range.rolling(window=20, min_periods=10).mean()
    vol_adj_momentum = data['close'].pct_change(5) / (atr_20d + 1e-8)
    
    # Moving Average Deviation
    sma_10d = data['close'].rolling(window=10, min_periods=5).mean()
    std_10d = data['close'].rolling(window=10, min_periods=5).std()
    ma_deviation = (data['close'] - sma_10d) / (std_10d + 1e-8)
    
    # Volume-Price Interaction
    # Volume-Weighted Return Reversal
    def volume_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    vol_slope_5d = data['volume'].rolling(window=5, min_periods=3).apply(volume_slope, raw=False)
    vol_weighted_reversal = ret_3d * vol_slope_5d
    
    # High-Low Range Volume Confirmation
    daily_range = (data['high'] - data['low']) / data['close']
    vol_10d_avg = data['volume'].rolling(window=10, min_periods=5).mean()
    vol_relative = data['volume'] / (vol_10d_avg + 1e-8)
    range_vol_confirmation = daily_range * vol_relative
    
    # Amount per Volume Trend
    amount_per_volume = data['amount'] / (data['volume'] + 1e-8)
    def ratio_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    amount_vol_trend = amount_per_volume.rolling(window=5, min_periods=3).apply(ratio_slope, raw=False)
    
    # Multi-Timeframe Signals
    # Momentum Consistency Score
    daily_returns = data['close'].pct_change(1)
    positive_returns = daily_returns.rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x > 0) / len(x), raw=False
    )
    
    # Return Variance Ratio
    ret_3d_rolling = data['close'].pct_change(1).rolling(window=3, min_periods=2)
    ret_10d_rolling = data['close'].pct_change(1).rolling(window=10, min_periods=5)
    var_ratio = ret_3d_rolling.var() / (ret_10d_rolling.var() + 1e-8)
    
    # Volume-Price Divergence
    price_return_5d = data['close'].pct_change(5)
    vol_return_5d = data['volume'].pct_change(5)
    vol_price_divergence = price_return_5d - vol_return_5d
    
    # Range-Based Factors
    # Opening Gap Persistence
    prev_close = data['close'].shift(1)
    opening_gap = (data['open'] - prev_close) / (prev_close + 1e-8)
    gap_persistence = opening_gap * np.sign(data['close'].pct_change(1))
    
    # Intraday Recovery Strength
    intraday_recovery = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    recovery_strength = intraday_recovery - intraday_recovery.shift(1)
    
    # Range Breakout Confirmation
    close_5d_high = data['close'] == data['close'].rolling(window=5, min_periods=3).max()
    close_5d_low = data['close'] == data['close'].rolling(window=5, min_periods=3).min()
    range_breakout = (close_5d_high.astype(int) - close_5d_low.astype(int)) * vol_relative
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'acceleration': acceleration,
        'vol_adj_momentum': vol_adj_momentum,
        'ma_deviation': ma_deviation,
        'vol_weighted_reversal': vol_weighted_reversal,
        'range_vol_confirmation': range_vol_confirmation,
        'amount_vol_trend': amount_vol_trend,
        'positive_returns': positive_returns,
        'var_ratio': var_ratio,
        'vol_price_divergence': vol_price_divergence,
        'gap_persistence': gap_persistence,
        'recovery_strength': recovery_strength,
        'range_breakout': range_breakout
    })
    
    # Z-score normalization for each factor
    normalized_factors = factors.apply(lambda x: (x - x.rolling(window=20, min_periods=10).mean()) / 
                                     (x.rolling(window=20, min_periods=10).std() + 1e-8))
    
    # Equal-weighted combination
    final_factor = normalized_factors.mean(axis=1)
    
    return final_factor
