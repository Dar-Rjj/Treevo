import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Price Fractal Divergence Factor
    # Volume entropy component
    volume_5d_std = df['volume'].rolling(window=5).std()
    volume_20d_std = df['volume'].rolling(window=20).std()
    volume_variance_ratio = volume_5d_std / volume_20d_std
    
    # 10-day volume autocorrelation
    volume_autocorr = df['volume'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )
    volume_entropy = volume_variance_ratio + volume_autocorr
    
    # Price fractal dimension component
    # Multi-timeframe range complexity
    range_5d = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()) / df['close']
    range_10d = (df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()) / df['close']
    range_20d = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close']
    range_complexity = (range_5d + range_10d + range_20d) / 3
    
    # Price path efficiency
    price_path_length = (df['high'] - df['low']).abs().rolling(window=10).sum()
    net_price_move = (df['close'] - df['close'].shift(10)).abs()
    price_efficiency = net_price_move / price_path_length
    price_fractal = range_complexity * price_efficiency
    
    # Divergence weighted by momentum and volume surge
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    volume_surge = df['volume'] / df['volume'].rolling(window=20).mean()
    divergence_weight = momentum_5d * volume_surge
    
    # Final Volume-Price Fractal Divergence Factor
    vp_fractal_divergence = volume_entropy * price_fractal * divergence_weight
    
    # Liquidity Absorption Momentum
    # Effective spread approximation
    effective_spread = (df['high'] - df['low']) / df['close']
    
    # Volume concentration (Gini-like measure)
    volume_rolling = df['volume'].rolling(window=10)
    volume_mean = volume_rolling.mean()
    volume_abs_diff = volume_rolling.apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
    )
    volume_concentration = volume_abs_diff / volume_mean
    
    spread_absorption = effective_spread * volume_concentration
    
    # Price impact and resilience
    price_impact = (df['close'] - df['open']) / df['open']
    price_resilience = 1 - (df['high'] - df['close']).abs() / (df['high'] - df['low'])
    liquidity_provision = price_impact * price_resilience
    
    # 3-day breakouts and trade size weighting
    breakout_3d = (df['close'] > df['high'].shift(3)).astype(int) - (df['close'] < df['low'].shift(3)).astype(int)
    directional_bias = breakout_3d * df['amount'] / df['amount'].rolling(window=10).mean()
    
    liquidity_momentum = spread_absorption + liquidity_provision + directional_bias
    
    # Volatility Regime Transition Detector
    # Volatility persistence and transitions
    volatility_5d = df['close'].pct_change().rolling(window=5).std()
    volatility_20d = df['close'].pct_change().rolling(window=20).std()
    vol_persistence = volatility_5d / volatility_20d
    
    # Transition probability (volatility changes)
    vol_change = volatility_5d.pct_change(5)
    transition_prob = vol_change.rolling(window=10).apply(
        lambda x: np.mean(np.abs(x) > np.std(x)), raw=False
    )
    
    # Momentum acceleration during transitions
    momentum_accel = (df['close'] / df['close'].shift(5) - 1) - (df['close'].shift(5) / df['close'].shift(10) - 1)
    
    # Volume pattern changes
    volume_pattern = df['volume'] / df['volume'].rolling(window=20).mean()
    volume_change = volume_pattern.pct_change(5)
    
    regime_transition = vol_persistence * transition_prob * momentum_accel * volume_change
    
    # Price Memory Decay Factor
    # Past return decay with exponential weighting
    returns_1d = df['close'].pct_change()
    returns_5d = df['close'].pct_change(5)
    returns_10d = df['close'].pct_change(10)
    
    decay_weights = np.array([0.5, 0.3, 0.2])  # Recent returns weighted more heavily
    weighted_returns = returns_1d * decay_weights[0] + returns_5d * decay_weights[1] + returns_10d * decay_weights[2]
    
    # Support/resistance persistence
    support_level = df['low'].rolling(window=20).min()
    resistance_level = df['high'].rolling(window=20).max()
    support_distance = (df['close'] - support_level) / (resistance_level - support_level)
    resistance_persistence = support_distance.rolling(window=10).std()
    
    # Pattern recognition and deviation
    ma_10 = df['close'].rolling(window=10).mean()
    ma_20 = df['close'].rolling(window=20).mean()
    pattern_deviation = (df['close'] - (ma_10 + ma_20) / 2) / df['close'].rolling(window=20).std()
    
    price_memory = weighted_returns * resistance_persistence * pattern_deviation
    
    # Momentum Quality Spectrum
    # Multi-timeframe consistency
    mom_5d = df['close'] / df['close'].shift(5) - 1
    mom_10d = df['close'] / df['close'].shift(10) - 1
    mom_20d = df['close'] / df['close'].shift(20) - 1
    momentum_consistency = (mom_5d * mom_10d * mom_20d).abs() ** (1/3)
    
    # Acceleration patterns
    acceleration = (mom_5d - mom_10d) - (mom_10d - mom_20d)
    
    # Risk-adjusted return (Sharpe-like)
    returns_20d = df['close'].pct_change().rolling(window=20)
    risk_adjusted = returns_20d.mean() / returns_20d.std()
    
    # Volume efficiency
    volume_efficiency = (df['close'].pct_change().abs() * df['amount']) / df['amount'].rolling(window=20).mean()
    
    momentum_quality = momentum_consistency * acceleration * risk_adjusted * volume_efficiency
    
    # Combine all factors with equal weighting
    combined_factor = (
        vp_fractal_divergence.fillna(0) +
        liquidity_momentum.fillna(0) +
        regime_transition.fillna(0) +
        price_memory.fillna(0) +
        momentum_quality.fillna(0)
    )
    
    return combined_factor
