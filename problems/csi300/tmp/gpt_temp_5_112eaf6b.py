import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Momentum Persistence with Volatility Clustering
    # Calculate Intraday Momentum Strength
    momentum_persistence = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Assess Volatility Clustering Pattern
    daily_ranges = df['high'] - df['low']
    range_autocorr = daily_ranges.rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) == 10 else np.nan, raw=False
    )
    volatility_clustering = range_autocorr
    
    # Combine Momentum and Volatility Signals
    intraday_return_sign = np.sign(df['close'] - df['open'])
    combined_signal = momentum_persistence * volatility_clustering
    directional_signal = combined_signal.abs() * intraday_return_sign
    
    # Add Volume Persistence Confirmation
    volume_changes = df['volume'].diff()
    volume_sign_consistency = volume_changes.rolling(window=5).apply(
        lambda x: (np.sign(x) == np.sign(x.iloc[-1])).sum() / len(x) if len(x) == 5 else np.nan, raw=False
    )
    
    factor1 = directional_signal * volume_sign_consistency
    
    # Price-Volume Divergence with Acceleration Detection
    # Calculate Price Acceleration
    returns = df['close'].pct_change()
    price_acceleration = returns.diff()
    
    # Measure Volume Divergence
    price_volume_divergence = (price_acceleration * df['volume']) / (df['amount'] + 1e-8)
    
    # Detect Regime Change
    current_range = df['high'] - df['low']
    historical_avg_range = (df['high'] - df['low']).rolling(window=10).mean().shift(1)
    volatility_regime = current_range / (historical_avg_range + 1e-8)
    
    factor2 = price_volume_divergence * volatility_regime * price_acceleration
    
    # Overnight Gap Fade with Intraday Range Confirmation
    # Calculate Overnight Gap Magnitude
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_magnitude = np.abs(overnight_gap)
    
    # Assess Intraday Range Efficiency
    range_efficiency = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Combine Gap and Efficiency
    gap_sign = np.sign(overnight_gap)
    gap_efficiency_combined = -gap_magnitude * range_efficiency * gap_sign
    
    # Add Volume Spike Confirmation
    median_volume = df['volume'].rolling(window=20).median().shift(1)
    volume_spike = df['volume'] / (median_volume + 1e-8)
    log_volume_spike = np.log1p(volume_spike)
    
    factor3 = gap_efficiency_combined * log_volume_spike
    
    # Multi-Timeframe Momentum Convergence
    # Calculate Short-Term Momentum
    short_slope = df['close'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) == 3 else np.nan, raw=False
    )
    
    # Calculate Medium-Term Momentum
    medium_slope = df['close'].rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) == 10 else np.nan, raw=False
    )
    
    # Detect Convergence Pattern
    momentum_convergence = short_slope / (medium_slope + 1e-8)
    
    # Add Volatility-Weighted Confirmation
    daily_range = df['high'] - df['low']
    
    factor4 = momentum_convergence * daily_range
    
    # Combine all factors with equal weighting
    final_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + factor4.fillna(0)) / 4
    
    return final_factor
