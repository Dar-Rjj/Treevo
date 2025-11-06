import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor using multiple technical approaches
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Factor 1: Momentum Decay Adjusted by Volume Skewness
    # Calculate 10-day and 20-day momentum
    momentum_10 = data['close'].pct_change(10)
    momentum_20 = data['close'].pct_change(20)
    
    # Calculate 15-day volume skewness
    volume_skew = data['volume'].rolling(window=15, min_periods=10).apply(
        lambda x: stats.skew(x) if len(x) >= 10 else 0, raw=True
    )
    volume_skew_sign = np.sign(volume_skew)
    
    # Apply exponential decay to momentum (0.9 decay factor)
    decay_factor = 0.9
    weights = np.array([decay_factor ** i for i in range(10)])
    weights = weights / weights.sum()
    
    decayed_momentum = momentum_10.rolling(window=10, min_periods=5).apply(
        lambda x: np.dot(x[-len(weights):], weights) if len(x) >= len(weights) else x.mean(),
        raw=True
    )
    
    factor1 = decayed_momentum * volume_skew_sign
    
    # Factor 2: Relative Strength Oscillator with Volume Confirmation
    # Calculate RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=7).mean()
    avg_loss = loss.rolling(window=14, min_periods=7).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_5_avg = data['volume'].rolling(window=5, min_periods=3).mean()
    vol_20_avg = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_ratio = vol_5_avg / vol_20_avg
    
    # Apply conditional logic
    factor2 = pd.Series(0, index=data.index)
    bullish_condition = (volume_ratio > 1.2) & (rsi < 30)
    bearish_condition = (volume_ratio > 1.2) & (rsi > 70)
    
    factor2 = factor2.where(~bullish_condition, 1)
    factor2 = factor2.where(~bearish_condition, -1)
    
    # Factor 3: Volatility Regime Adaptive Factor
    # Calculate Average True Range
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=10, min_periods=5).mean()
    
    # Classify volatility regime (median split)
    vol_threshold = atr.rolling(window=50, min_periods=25).median()
    high_vol_regime = atr > vol_threshold
    
    # High volatility: mean reversion
    ret_5 = data['close'].pct_change(5)
    mean_reversion = -ret_5 / (atr + 1e-8)
    
    # Low volatility: momentum
    momentum_20 = data['close'].pct_change(20)
    momentum_scaled = momentum_20 / (atr + 1e-8)
    
    # Combine based on regime
    factor3 = pd.Series(0, index=data.index)
    factor3[high_vol_regime] = mean_reversion[high_vol_regime]
    factor3[~high_vol_regime] = momentum_scaled[~high_vol_regime]
    
    # Factor 4: Intraday Pressure Index
    # Calculate intraday pressure
    high_low_range = data['high'] - data['low']
    # Handle zero range cases
    high_low_range = high_low_range.replace(0, np.nan)
    intraday_pressure = (data['close'] - data['open']) / high_low_range
    intraday_pressure = intraday_pressure.fillna(0)
    
    # Volume-weighted accumulation
    pressure_volume = intraday_pressure * data['volume']
    factor4 = pressure_volume.rolling(window=5, min_periods=3).sum()
    
    # Factor 5: Liquidity-Adjusted Price Trend
    # Calculate linear regression slope for price trend
    def linear_trend(y):
        if len(y) < 2:
            return 0
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    price_trend = data['close'].rolling(window=15, min_periods=10).apply(
        linear_trend, raw=True
    )
    
    # Calculate liquidity proxy
    avg_trade_size = data['amount'] / (data['volume'] + 1e-8)
    liquidity_score = 1 / (avg_trade_size.rolling(window=10, min_periods=5).std() + 1e-8)
    
    # Combine trend and liquidity
    factor5 = price_trend * liquidity_score
    factor5 = factor5.rolling(window=3, min_periods=2).mean()
    
    # Factor 6: Volume-Cluster Breakout Detector
    # Identify volume clusters (abnormal spikes)
    vol_median = data['volume'].rolling(window=50, min_periods=25).median()
    vol_std = data['volume'].rolling(window=50, min_periods=25).std()
    volume_spike = data['volume'] > (vol_median + 2 * vol_std)
    
    # Price breakout analysis during volume clusters
    rolling_high = data['high'].rolling(window=20, min_periods=10).max()
    rolling_low = data['low'].rolling(window=20, min_periods=10).min()
    
    new_high_break = (data['high'] == rolling_high) & volume_spike
    new_low_break = (data['low'] == rolling_low) & volume_spike
    
    # Breakout strength
    breakout_strength_high = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    breakout_strength_low = (data['open'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    
    factor6 = pd.Series(0, index=data.index)
    factor6[new_high_break & (breakout_strength_high > 0.6)] = 1
    factor6[new_low_break & (breakout_strength_low > 0.6)] = -1
    
    # Factor 7: Amplitude-Volume Divergence Factor
    # Calculate price amplitude
    daily_range = data['high'] - data['low']
    amplitude_avg = daily_range.rolling(window=10, min_periods=7).mean()
    
    # Volume pattern
    volume_avg = data['volume'].rolling(window=10, min_periods=7).mean()
    volume_trend = volume_avg.pct_change(3)
    
    # Detect divergence
    high_amplitude = daily_range > amplitude_avg
    decreasing_volume = volume_trend < 0
    low_amplitude = daily_range < amplitude_avg * 0.7
    increasing_volume = volume_trend > 0.1
    
    factor7 = pd.Series(0, index=data.index)
    factor7[high_amplitude & decreasing_volume] = -1  # Reversal signal
    factor7[low_amplitude & increasing_volume] = 1   # Breakout signal
    
    # Combine all factors with equal weighting
    factors = [factor1, factor2, factor3, factor4, factor5, factor6, factor7]
    
    # Standardize each factor
    standardized_factors = []
    for factor in factors:
        if factor.notna().any():
            mean = factor.rolling(window=100, min_periods=50).mean()
            std = factor.rolling(window=100, min_periods=50).std()
            standardized = (factor - mean) / (std + 1e-8)
            standardized_factors.append(standardized)
    
    # Equal weighted combination
    if standardized_factors:
        combined_alpha = pd.concat(standardized_factors, axis=1).mean(axis=1)
    else:
        combined_alpha = pd.Series(0, index=data.index)
    
    return combined_alpha
