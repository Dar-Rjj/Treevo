import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Volatility-Adjusted Turnover Momentum
    # 5-day sum of Amount / Close
    turnover_5d = (data['amount'] / data['close']).rolling(window=5, min_periods=5).sum()
    
    # 20-day average of |High - Low|/Close
    vol_20d = ((data['high'] - data['low']) / data['close']).abs().rolling(window=20, min_periods=20).mean()
    
    # Volatility-normalized turnover
    vol_normalized_turnover = turnover_5d / vol_20d
    
    # 3-day cumulative sum of Volume * sign(Close - Open)
    directional_volume = (data['volume'] * np.sign(data['close'] - data['open'])).rolling(window=3, min_periods=3).sum()
    
    # Apply inverse hyperbolic sine transformation
    directional_persistence = np.arcsinh(directional_volume)
    
    # Combine components
    factor1 = vol_normalized_turnover * directional_persistence
    
    # Pressure-Duration Divergence
    # 5-day sum of (High - Low) * Volume / Amount
    pressure_5d = ((data['high'] - data['low']) * data['volume'] / data['amount']).rolling(window=5, min_periods=5).sum()
    
    # 10-day sum of (High - Low) * Volume / Amount
    pressure_10d = ((data['high'] - data['low']) * data['volume'] / data['amount']).rolling(window=10, min_periods=10).sum()
    
    # Duration-weighted price pressure
    duration_pressure = pressure_5d - pressure_10d
    
    # |Open - Close_prev| / (High_prev - Low_prev)
    prev_close = data['close'].shift(1)
    prev_high = data['high'].shift(1)
    prev_low = data['low'].shift(1)
    gap_efficiency = (data['open'] - prev_close).abs() / (prev_high - prev_low).replace(0, np.nan)
    
    # Current volume percentile (20-day)
    volume_percentile = data['volume'].rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Overnight gap efficiency scaled by volume percentile
    gap_efficiency_scaled = gap_efficiency * volume_percentile
    
    # Combine components
    factor2 = duration_pressure * gap_efficiency_scaled
    
    # Range-Constrained Momentum Acceleration
    # (Close - Close_5) / (20-day max(High) - 20-day min(Low))
    close_5 = data['close'].shift(5)
    high_20max = data['high'].rolling(window=20, min_periods=20).max()
    low_20min = data['low'].rolling(window=20, min_periods=20).min()
    vol_bounded_momentum = (data['close'] - close_5) / (high_20max - low_20min).replace(0, np.nan)
    
    # 5-day correlation between Volume and |Close - Open|
    volume_abs_change_corr = data['volume'].rolling(window=5, min_periods=5).corr(
        (data['close'] - data['open']).abs()
    )
    
    # Apply Fisher transformation
    def fisher_transform(x):
        x_clipped = np.clip(x, -0.999, 0.999)
        return 0.5 * np.log((1 + x_clipped) / (1 - x_clipped))
    
    volume_trend_consistency = volume_abs_change_corr.apply(fisher_transform)
    
    # Volatility-bounded momentum with volume trend consistency
    momentum_with_volume = vol_bounded_momentum * volume_trend_consistency
    
    # (Close - Low) / (High - Low) - 0.5
    intraday_range_util = ((data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)) - 0.5
    
    # 3-day sum of sign(Open - Close_prev)
    gap_persistence = np.sign(data['open'] - prev_close).rolling(window=3, min_periods=3).sum()
    
    # Intraday range utilization scaled by gap persistence
    range_util_scaled = intraday_range_util * gap_persistence
    
    # Combine components
    factor3 = momentum_with_volume * range_util_scaled
    
    # Liquidity-Adjusted Reversal Signal
    # 3-day sum of (Close - Open) * Volume / Amount
    reversal_strength = ((data['close'] - data['open']) * data['volume'] / data['amount']).rolling(window=3, min_periods=3).sum()
    
    # 10-day average of |Close - Open| * Volume / Amount
    reversal_magnitude = ((data['close'] - data['open']).abs() * data['volume'] / data['amount']).rolling(window=10, min_periods=10).mean()
    
    # Volume-weighted reversal strength
    volume_weighted_reversal = reversal_strength / reversal_magnitude.replace(0, np.nan)
    
    # Current 5-day volatility / 20-day volatility
    vol_5d = data['close'].pct_change().rolling(window=5, min_periods=5).std()
    vol_20d_long = data['close'].pct_change().rolling(window=20, min_periods=20).std()
    vol_regime = vol_5d / vol_20d_long.replace(0, np.nan)
    
    # Apply logistic transformation
    def logistic_transform(x):
        return 1 / (1 + np.exp(-x))
    
    vol_regime_adjusted = vol_regime.apply(logistic_transform)
    
    # Combine components
    factor4 = volume_weighted_reversal * vol_regime_adjusted
    
    # Turnover-Concentration Momentum
    # 5-day sum of Amount when (Close - Open)/Open > 1%
    up_days_mask = ((data['close'] - data['open']) / data['open']) > 0.01
    up_turnover = (data['amount'] * up_days_mask).rolling(window=5, min_periods=5).sum()
    
    # 5-day sum of Amount when (Close - Open)/Open < -1%
    down_days_mask = ((data['close'] - data['open']) / data['open']) < -0.01
    down_turnover = (data['amount'] * down_days_mask).rolling(window=5, min_periods=5).sum()
    
    # High-low turnover concentration
    turnover_concentration = up_turnover / down_turnover.replace(0, np.nan)
    
    # 3-day cumulative product of (High - Low)/(High_prev - Low_prev)
    range_ratio = (data['high'] - data['low']) / (prev_high - prev_low).replace(0, np.nan)
    range_expansion = range_ratio.rolling(window=3, min_periods=3).apply(lambda x: x.prod(), raw=False)
    
    # Current volume acceleration (5-day/10-day volume ratio)
    vol_5d_avg = data['volume'].rolling(window=5, min_periods=5).mean()
    vol_10d_avg = data['volume'].rolling(window=10, min_periods=10).mean()
    volume_acceleration = vol_5d_avg / vol_10d_avg.replace(0, np.nan)
    
    # Range expansion persistence scaled by volume acceleration
    range_persistence_scaled = range_expansion * volume_acceleration
    
    # Combine components
    factor5 = turnover_concentration * range_persistence_scaled
    
    # Efficiency-Weighted Gap Signal
    # (Open - Close_prev) / Close_prev
    gap_signal = (data['open'] - prev_close) / prev_close
    
    # Multiply by Volume / 20-day average Volume
    vol_20d_avg = data['volume'].rolling(window=20, min_periods=20).mean()
    gap_volume_adjusted = gap_signal * (data['volume'] / vol_20d_avg.replace(0, np.nan))
    
    # 3-day average of |Close - Open| / (High - Low)
    intraday_efficiency = ((data['close'] - data['open']).abs() / (data['high'] - data['low']).replace(0, np.nan)).rolling(window=3, min_periods=3).mean()
    
    # Apply square root transformation
    efficiency_persistence = intraday_efficiency.apply(lambda x: np.sqrt(abs(x)) * np.sign(x) if pd.notnull(x) else x)
    
    # Combine components
    factor6 = gap_volume_adjusted * efficiency_persistence
    
    # Combine all factors with equal weights
    combined_factor = (factor1 + factor2 + factor3 + factor4 + factor5 + factor6) / 6
    
    # Return the combined factor series
    return combined_factor
