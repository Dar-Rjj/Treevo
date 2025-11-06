import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Asymmetric Gap-Driven Momentum
    # Overnight Gap Component
    overnight_return = df['open'] / df['close'].shift(1) - 1
    positive_overnight = overnight_return.clip(lower=0)
    asymmetric_gap = positive_overnight ** 2
    
    # Intraday Momentum Component
    intraday_return = df['close'] / df['open'] - 1
    median_volume = df['volume'].rolling(window=15, min_periods=1).median()
    volume_ratio = df['volume'] / median_volume
    intraday_strength = intraday_return * volume_ratio
    
    # Combine components
    gap_momentum = asymmetric_gap * intraday_strength
    
    # Volatility-Clustered Reversal
    # Volatility Clustering
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    range_autocorr = daily_range.rolling(window=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    regime_indicator = np.where(range_autocorr > 0.3, 1, -1)
    
    # Price Reversal
    two_day_return = df['close'].shift(-1) / df['close'].shift(1) - 1
    reversal_return = -two_day_return  # Reverse sign for mean reversion
    
    # Combine with regime
    clustered_reversal = reversal_return * regime_indicator
    
    # Volume-Implied Liquidity Shock
    # Volume Fractality (simplified Hurst approximation)
    volume_changes = df['volume'].pct_change()
    
    def hurst_approximation(series):
        if len(series) < 2:
            return 0.5
        lags = range(2, min(6, len(series)))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    hurst_exp = volume_changes.rolling(window=10, min_periods=2).apply(
        hurst_approximation, raw=False
    )
    
    # Volume deviation
    volume_mean = df['volume'].rolling(window=20, min_periods=1).mean()
    volume_std = df['volume'].rolling(window=20, min_periods=1).std()
    volume_deviation = (df['volume'] - volume_mean) / volume_std
    
    # Shock indicator
    shock_indicator = np.where((hurst_exp < 0.4) & (volume_deviation > 2), 1, 0)
    
    # Price efficiency
    intraday_range = df['high'] - df['low']
    daily_return_abs = abs(df['close'] / df['open'] - 1)
    price_efficiency = intraday_range / (df['close'] * daily_return_abs).replace(0, np.nan)
    price_efficiency = price_efficiency.fillna(0)
    
    # Combine components
    liquidity_shock = shock_indicator * price_efficiency
    
    # Momentum-Decay Acceleration
    # Momentum duration
    returns = df['close'].pct_change()
    sign_changes = (returns * returns.shift(1)) < 0
    duration = sign_changes.groupby(sign_changes.cumsum()).cumcount() + 1
    
    # Decay factor
    decay_factor = np.exp(-duration / 5.0)  # Exponential decay with half-life of 5 days
    
    # Raw momentum (market neutral residual approach)
    market_return = df['close'].pct_change().rolling(window=5).mean()
    raw_momentum = df['close'].pct_change(periods=5) - market_return
    
    # Apply decay
    decayed_momentum = raw_momentum * decay_factor
    
    # Range-Expansion Breakout
    # Normalized range
    daily_range_abs = df['high'] - df['low']
    avg_range = daily_range_abs.rolling(window=20, min_periods=1).mean()
    normalized_range = daily_range_abs / avg_range
    
    # Volume condition
    avg_volume_10d = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_condition = df['volume'] > 1.2 * avg_volume_10d
    
    # Expansion events
    expansion_events = (normalized_range > 1.5) & volume_condition
    
    # Directional signal
    range_midpoint = (df['high'] + df['low']) / 2
    directional_signal = np.where(df['close'] > range_midpoint, 1, -1)
    
    # Combine components
    range_breakout = normalized_range * directional_signal * expansion_events
    
    # Combine all factors with equal weights
    combined_factor = (
        gap_momentum.fillna(0) +
        clustered_reversal.fillna(0) +
        liquidity_shock.fillna(0) +
        decayed_momentum.fillna(0) +
        range_breakout.fillna(0)
    )
    
    return combined_factor
