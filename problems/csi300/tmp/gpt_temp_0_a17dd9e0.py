import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Nonlinear Price-Volume Divergence
    # Daily Price Range
    daily_range = data['high'] - data['low']
    
    # Volume-Adjusted Range
    volume_adjusted_range = daily_range * data['volume']
    
    # Price Momentum (5-day vs 10-day return)
    ret_5d = data['close'].pct_change(5)
    ret_10d = data['close'].pct_change(10)
    price_momentum = ret_5d - ret_10d
    
    # Detect Divergence (momentum vs volume pattern)
    volume_ma_5 = data['volume'].rolling(window=5).mean()
    volume_ma_10 = data['volume'].rolling(window=10).mean()
    volume_momentum = volume_ma_5 / volume_ma_10 - 1
    
    divergence = price_momentum * volume_momentum
    
    # Volatility Regime Factor
    # Identify Volatility Periods (20-day std vs 60-day median)
    vol_20d = data['close'].pct_change().rolling(window=20).std()
    vol_60d_median = data['close'].pct_change().rolling(window=60).std().rolling(window=20).median()
    volatility_regime = vol_20d / vol_60d_median
    
    # Measure Volume Persistence (5-day autocorrelation)
    volume_autocorr = data['volume'].rolling(window=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )
    
    # Combine Signals
    volatility_signal = volatility_regime * volume_autocorr
    
    # Liquidity Gap Alpha
    # Estimate Spread Proxy
    spread_proxy = (data['high'] - data['low']) / data['close']
    
    # Measure Volume Concentration (intraday distribution)
    # Using amount/volume as proxy for average price, then calculate concentration
    avg_price = data['amount'] / data['volume']
    price_range_ratio = (data['high'] - avg_price) / (data['high'] - data['low']).replace(0, np.nan)
    volume_concentration = 1 - 2 * abs(price_range_ratio - 0.5)
    
    # Generate Signal
    liquidity_signal = spread_proxy * volume_concentration
    
    # Momentum Acceleration
    # Calculate Acceleration (2nd derivative of 5-day MA)
    ma_5 = data['close'].rolling(window=5).mean()
    ma_5_velocity = ma_5.diff()
    ma_5_acceleration = ma_5_velocity.diff()
    
    # Analyze Volume Trend (MA crossover)
    volume_ma_fast = data['volume'].rolling(window=3).mean()
    volume_ma_slow = data['volume'].rolling(window=8).mean()
    volume_trend = volume_ma_fast / volume_ma_slow - 1
    
    # Generate Signal
    momentum_signal = ma_5_acceleration * volume_trend
    
    # Structural Break Detection
    # Identify Regime Changes (variance ratio)
    returns = data['close'].pct_change()
    var_short = returns.rolling(window=10).var()
    var_long = returns.rolling(window=30).var()
    variance_ratio = var_short / var_long
    
    # Analyze Volume Clustering (abnormal concentration)
    volume_zscore = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    volume_clustering = abs(volume_zscore)
    
    # Generate Signal
    structural_signal = variance_ratio * volume_clustering
    
    # Combine all factors with equal weights
    factor = (
        0.2 * divergence.fillna(0) +
        0.2 * volatility_signal.fillna(0) +
        0.2 * liquidity_signal.fillna(0) +
        0.2 * momentum_signal.fillna(0) +
        0.2 * structural_signal.fillna(0)
    )
    
    return factor
