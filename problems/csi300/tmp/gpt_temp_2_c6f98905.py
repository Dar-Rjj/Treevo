import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate multiple alpha factors using OHLCV data with no future information.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Dynamic Volatility-Adjusted Price Momentum
    # Rolling window parameters
    N1 = 20  # momentum and volatility lookback
    
    # Price momentum
    price_momentum = (data['close'] - data['close'].shift(N1)) / data['close'].shift(N1)
    
    # Dynamic volatility using high-low range
    daily_range = (data['high'] - data['low']) / data['close']
    range_volatility = daily_range.rolling(window=N1).std()
    avg_range = daily_range.rolling(window=N1).mean()
    dynamic_vol = range_volatility / avg_range.replace(0, np.nan)
    
    # Volatility-adjusted momentum
    factor1 = price_momentum / dynamic_vol.replace(0, np.nan)
    
    # Factor 2: Volume-Implied Order Imbalance
    # Intraday price pressure
    intraday_pressure = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume trend
    M = 10  # volume lookback
    volume_ma = data['volume'].rolling(window=M).mean()
    volume_trend = data['volume'] / volume_ma.replace(0, np.nan)
    
    # Volume-confirmed price pressure
    factor2 = intraday_pressure * volume_trend
    
    # Factor 3: Amplitude-Frequency Price Oscillation
    K = 10  # oscillation frequency period
    
    # Price oscillation frequency
    price_changes = data['close'].diff()
    directional_changes = ((price_changes > 0) & (price_changes.shift(1) < 0)) | ((price_changes < 0) & (price_changes.shift(1) > 0))
    oscillation_freq = directional_changes.rolling(window=K).sum() / K
    
    # Average True Range
    N3 = 14  # ATR period
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=N3).mean()
    
    # Comprehensive oscillation factor
    factor3 = oscillation_freq * atr
    
    # Factor 4: Liquidity-Adjusted Return Persistence
    M4 = 20  # autocorrelation period
    
    # Return autocorrelation
    returns = data['close'].pct_change()
    autocorr = returns.rolling(window=M4).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # Relative liquidity (volume-to-amount ratio trend)
    volume_amount_ratio = data['volume'] / data['amount'].replace(0, np.nan)
    liquidity_trend = volume_amount_ratio / volume_amount_ratio.rolling(window=M4).mean().replace(0, np.nan)
    
    # Liquidity-conditioned persistence
    factor4 = autocorr * liquidity_trend
    
    # Factor 5: Volatility-Regime Adaptive Momentum
    # Volatility regime detection
    range_vol = daily_range.rolling(window=20).std()
    vol_percentile = range_vol.rolling(window=50).rank(pct=True)
    
    # Regime-specific momentum
    short_momentum = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    long_momentum = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    
    # Adaptive momentum based on volatility regime
    high_vol_regime = (vol_percentile > 0.7)
    low_vol_regime = (vol_percentile < 0.3)
    
    regime_momentum = pd.Series(index=data.index, dtype=float)
    regime_momentum[high_vol_regime] = short_momentum[high_vol_regime] * 0.7  # dampening
    regime_momentum[low_vol_regime] = long_momentum[low_vol_regime] * 1.3    # amplification
    regime_momentum[~high_vol_regime & ~low_vol_regime] = long_momentum[~high_vol_regime & ~low_vol_regime]
    
    factor5 = regime_momentum
    
    # Factor 6: Volume-Weighted Price Acceleration
    # Price acceleration (second derivative)
    price_velocity = data['close'].pct_change()
    price_acceleration = price_velocity.diff()
    
    # Volume-weighted acceleration
    avg_volume = data['volume'].rolling(window=10).mean()
    volume_weighted_accel = price_acceleration * data['volume'] / avg_volume.replace(0, np.nan)
    
    factor6 = volume_weighted_accel
    
    # Factor 7: Intraday Range Persistence Factor
    # Range autocorrelation
    range_autocorr = daily_range.rolling(window=15).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # Price trend alignment
    price_trend = data['close'].rolling(window=10).apply(lambda x: (x[-1] - x[0]) / x[0], raw=True)
    
    # Range-price alignment
    range_price_alignment = range_autocorr * price_trend
    
    factor7 = range_price_alignment
    
    # Factor 8: Momentum-Volume Divergence Factor
    # Momentum and volume trends
    price_momentum_trend = data['close'].pct_change(periods=5)
    volume_momentum_trend = data['volume'].pct_change(periods=5)
    
    # Divergence detection
    rising_price_falling_volume = (price_momentum_trend > 0) & (volume_momentum_trend < 0)
    falling_price_rising_volume = (price_momentum_trend < 0) & (volume_momentum_trend > 0)
    
    # Divergence signal
    divergence_signal = pd.Series(0, index=data.index)
    divergence_signal[rising_price_falling_volume] = -1  # bearish divergence
    divergence_signal[falling_price_rising_volume] = 1   # bullish divergence
    
    factor8 = divergence_signal
    
    # Combine factors (equal weight for demonstration)
    combined_factor = (
        factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
        factor4.fillna(0) + factor5.fillna(0) + factor6.fillna(0) + 
        factor7.fillna(0) + factor8.fillna(0)
    ) / 8
    
    return combined_factor
