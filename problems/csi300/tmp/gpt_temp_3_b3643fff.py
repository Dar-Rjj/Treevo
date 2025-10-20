import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators:
    - Momentum-Adjusted Volatility Breakout
    - Volume-Weighted Price Acceleration  
    - Intraday Pressure Cumulation
    - Relative Strength Convergence
    - Liquidity-Adjusted Trend Persistence
    - Asymmetric Volume Response
    - Range Expansion Momentum
    - Volume-Cluster Identification
    """
    
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Momentum-Adjusted Volatility Breakout
    # Compute rolling volatility from daily High-Low range
    hl_range = data['high'] - data['low']
    vol_20d = hl_range.rolling(window=20).std()
    
    # Identify breakout days when range exceeds historical volatility
    breakout = (hl_range > vol_20d * 1.5).astype(int)
    
    # Calculate short-term momentum using Close prices
    momentum_5d = data['close'].pct_change(5)
    
    # Combine breakout strength with momentum direction
    breakout_factor = breakout * momentum_5d * hl_range / data['close']
    
    # 2. Volume-Weighted Price Acceleration
    # Calculate price acceleration as second derivative of Close
    price_velocity = data['close'].pct_change()
    price_acceleration = price_velocity.diff()
    
    # Apply volume weighting using Volume data
    volume_weighted_accel = price_acceleration * data['volume'] / data['volume'].rolling(20).mean()
    
    # Scale by historical volatility for market adjustment
    vol_adj_accel = volume_weighted_accel / data['close'].pct_change().rolling(20).std()
    
    # 3. Intraday Pressure Cumulation
    # Calculate buying/selling pressure from OHLC
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    pressure = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 0.001)
    
    # Accumulate pressure with decay over multiple days
    decay_factor = 0.9
    cum_pressure = pressure.copy()
    for i in range(1, len(cum_pressure)):
        cum_pressure.iloc[i] = pressure.iloc[i] + decay_factor * cum_pressure.iloc[i-1]
    
    # Detect extremes for mean reversion signals
    pressure_zscore = (cum_pressure - cum_pressure.rolling(20).mean()) / cum_pressure.rolling(20).std()
    pressure_factor = -pressure_zscore  # Negative for mean reversion
    
    # 4. Relative Strength Convergence
    # Compute relative strength vs market using rolling performance
    market_perf = data['close'].pct_change(10).rolling(5).mean()
    stock_perf = data['close'].pct_change(5)
    relative_strength = stock_perf - market_perf
    
    # Track convergence/divergence of short vs long-term performance
    short_term_rs = relative_strength.rolling(5).mean()
    long_term_rs = relative_strength.rolling(20).mean()
    convergence = short_term_rs - long_term_rs
    
    # Generate directional signal from strength and convergence
    rs_factor = np.sign(relative_strength) * convergence
    
    # 5. Liquidity-Adjusted Trend Persistence
    # Identify primary trend using multiple timeframes
    trend_short = data['close'].rolling(5).mean()
    trend_medium = data['close'].rolling(20).mean()
    trend_long = data['close'].rolling(50).mean()
    
    # Measure trend strength
    trend_alignment = ((trend_short > trend_medium) & (trend_medium > trend_long)).astype(int) - \
                     ((trend_short < trend_medium) & (trend_medium < trend_long)).astype(int)
    
    # Measure liquidity conditions from Volume and Amount
    volume_trend = data['volume'] / data['volume'].rolling(20).mean()
    amount_trend = data['amount'] / data['amount'].rolling(20).mean()
    liquidity_score = (volume_trend + amount_trend) / 2
    
    # Evaluate trend quality combining trend strength and liquidity
    trend_persistence = trend_alignment * liquidity_score
    
    # 6. Asymmetric Volume Response
    # Calculate return-volume relationship using Close returns and Volume
    returns = data['close'].pct_change()
    volume_response = data['volume'].pct_change()
    
    # Measure asymmetry in volume response to up vs down moves
    up_volume_corr = returns[returns > 0].rolling(10).corr(volume_response[returns > 0])
    down_volume_corr = returns[returns < 0].rolling(10).corr(volume_response[returns < 0])
    volume_asymmetry = up_volume_corr - down_volume_corr
    
    # Detect regime changes from volume behavior shifts
    volume_regime = volume_asymmetry.rolling(5).std()
    volume_factor = volume_asymmetry / volume_regime.replace(0, 1)
    
    # 7. Range Expansion Momentum
    # Compute normalized range using High-Low vs historical average
    range_norm = hl_range / hl_range.rolling(20).mean()
    
    # Measure momentum from Close position within daily range
    range_position = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, 1)
    range_momentum = range_position.rolling(5).mean()
    
    # Combine range expansion with momentum for breakout confirmation
    range_factor = range_norm * range_momentum
    
    # 8. Volume-Cluster Identification
    # Detect volume anomalies from Volume data
    volume_zscore = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
    volume_anomaly = (volume_zscore > 2).astype(int)
    
    # Analyze price behavior around volume clusters using OHLC
    volume_cluster_returns = data['close'].pct_change(3).shift(-3)  # Future returns for pattern detection
    volume_pattern = volume_anomaly * data['close'].pct_change(1)
    
    # Generate predictive signals from volume-based patterns
    volume_signal = volume_pattern.rolling(10).mean()
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'breakout': breakout_factor,
        'acceleration': vol_adj_accel,
        'pressure': pressure_factor,
        'relative_strength': rs_factor,
        'trend': trend_persistence,
        'volume_asymmetry': volume_factor,
        'range': range_factor,
        'volume_cluster': volume_signal
    })
    
    # Normalize each factor
    normalized_factors = factors.apply(lambda x: (x - x.rolling(50).mean()) / x.rolling(50).std())
    
    # Combine with equal weights
    combined_factor = normalized_factors.mean(axis=1)
    
    return combined_factor
