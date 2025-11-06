import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Regime Adaptive Price-Volume Divergence
    # Compute volatility regime classification
    df['high_low_vol_5d'] = (df['high'] - df['low']).rolling(window=5).std()
    df['close_vol_20d'] = df['close'].pct_change().rolling(window=20).std()
    
    # Detect price-volume correlation shifts
    df['price_vol_corr_3d'] = df['close'].pct_change().rolling(window=3).corr(df['volume'].pct_change())
    df['price_vol_corr_10d'] = df['close'].pct_change().rolling(window=10).corr(df['volume'].pct_change())
    
    # Generate regime-weighted signal
    volatility_ratio = df['high_low_vol_5d'] / (df['close_vol_20d'] + 1e-8)
    correlation_divergence = df['price_vol_corr_3d'] - df['price_vol_corr_10d']
    price_return_3d = df['close'].pct_change(3).abs()
    
    signal_1 = correlation_divergence * volatility_ratio * price_return_3d
    
    # Asymmetric Volume-Weighted Gap Efficiency
    # Calculate directional gap efficiency
    gap = df['open'] / df['close'].shift(1) - 1
    upward_gap_efficiency = np.where(gap > 0, (df['high'] - df['open']) / (df['open'] - df['low'].shift(1) + 1e-8), 0)
    downward_gap_efficiency = np.where(gap < 0, (df['open'] - df['low']) / (df['high'].shift(1) - df['open'] + 1e-8), 0)
    
    # Analyze volume asymmetry
    volume_quantile = df['volume'].rolling(window=5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    volume_concentration = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Generate net directional signal
    volume_asymmetry = volume_quantile * volume_concentration
    net_gap_efficiency = upward_gap_efficiency - downward_gap_efficiency
    signal_2 = net_gap_efficiency * volume_asymmetry * gap.abs()
    
    # Multi-Scale Liquidity Momentum Convergence
    # Compute liquidity momentum
    turnover = df['amount'] / df['close']
    turnover_accel_2d = turnover.pct_change(2)
    turnover_momentum_5d = turnover.pct_change(5)
    
    # Assess convergence patterns
    momentum_alignment = np.sign(turnover_accel_2d) == np.sign(turnover_momentum_5d)
    convergence_strength = (turnover_accel_2d.abs() + turnover_momentum_5d.abs()) / 2
    
    # Generate price prediction signal
    price_range_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    volume_trend = df['volume'].pct_change(3)
    signal_3 = convergence_strength * momentum_alignment * price_range_efficiency * volume_trend
    
    # Regime-Dependent Reversal Intensity
    # Identify microstructure regimes
    range_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    volume_clustering = df['volume'].rolling(window=5).std() / (df['volume'].rolling(window=5).mean() + 1e-8)
    
    # Detect reversal patterns
    price_reversal = -df['close'].pct_change(2) * df['close'].pct_change(1)
    volume_reversal = df['volume'].pct_change(2) * df['volume'].pct_change(1)
    
    # Generate regime-responsive signal
    regime_threshold = range_efficiency.rolling(window=10).quantile(0.7)
    regime_weight = np.where(range_efficiency > regime_threshold, 1.5, 0.7)
    preceding_momentum = df['close'].pct_change(3)
    signal_4 = price_reversal * volume_reversal * regime_weight * preceding_momentum
    
    # Cross-Sectional Relative Breakout Strength
    # Calculate breakout metrics
    price_breakout = (df['close'] - df['low'].rolling(window=15).min()) / (df['high'].rolling(window=15).max() - df['low'].rolling(window=15).min() + 1e-8)
    volume_breakout = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-8)
    
    # Generate relative rankings (using rolling rank for cross-sectional simulation)
    breakout_strength = price_breakout * volume_breakout
    relative_rank = breakout_strength.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    relative_momentum = df['close'].pct_change(5)
    
    # Create adaptive signal
    market_regime = df['close'].pct_change(10).abs()
    mean_reversion_decay = np.exp(-breakout_strength)
    signal_5 = relative_rank * relative_momentum * market_regime * mean_reversion_decay
    
    # Temporal Pattern Recognition in Price-Volume Dynamics
    # Analyze pattern persistence
    open_close_consistency = (df['close'] - df['open']).rolling(window=3).apply(lambda x: np.sign(x).sum() / 3, raw=False)
    high_low_capture = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Detect sequential patterns
    price_sequence = df['close'].pct_change().rolling(window=3).apply(lambda x: len(set(np.sign(x))) == 1, raw=False)
    volume_sequence = df['volume'].pct_change().rolling(window=3).apply(lambda x: len(set(np.sign(x))) == 1, raw=False)
    
    # Generate pattern-based signals
    pattern_strength = price_sequence.astype(float) * volume_sequence.astype(float)
    momentum_confirmation = df['close'].pct_change(2)
    signal_6 = pattern_strength * open_close_consistency * high_low_capture * momentum_confirmation
    
    # Combine all signals with equal weights
    combined_signal = (signal_1 + signal_2 + signal_3 + signal_4 + signal_5 + signal_6) / 6
    
    return combined_signal
