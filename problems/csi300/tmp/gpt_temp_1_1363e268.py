import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Weighted Gap Efficiency Momentum factor
    """
    data = df.copy()
    
    # Calculate Gap-Efficiency Components
    # Overnight Gap Momentum
    data['raw_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_momentum'] = data['raw_gap'].ewm(span=5, adjust=False).mean()
    data['gap_acceleration'] = data['gap_momentum'] - data['gap_momentum'].shift(3)
    
    # Efficiency Pressure
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, np.nan)  # Avoid division by zero
    
    data['buying_pressure'] = ((2 * data['close'] - data['low'] - data['high']) / high_low_range) * data['volume']
    data['selling_pressure'] = ((data['high'] + data['low'] - 2 * data['close']) / high_low_range) * data['volume']
    data['net_pressure'] = (data['buying_pressure'] - data['selling_pressure']).ewm(span=5, adjust=False).mean()
    
    # Fractal Efficiency
    close_diff_abs = data['close'].diff().abs()
    data['efficiency'] = (data['close'] - data['close'].shift(10)) / close_diff_abs.rolling(window=10).sum()
    data['efficiency_momentum'] = data['efficiency'] - data['efficiency'].shift(5)
    data['efficiency_acceleration'] = data['efficiency_momentum'] - data['efficiency_momentum'].shift(3)
    
    # Generate Gap-Efficiency Convergence
    # Gap-Efficiency Alignment
    gap_eff_corr = data['gap_momentum'].rolling(window=8).corr(data['efficiency_momentum'])
    
    # Directional consistency
    gap_dir_5 = np.sign(data['gap_momentum'] - data['gap_momentum'].shift(5))
    gap_dir_8 = np.sign(data['gap_momentum'] - data['gap_momentum'].shift(8))
    gap_dir_12 = np.sign(data['gap_momentum'] - data['gap_momentum'].shift(12))
    
    eff_dir_5 = np.sign(data['efficiency_momentum'] - data['efficiency_momentum'].shift(5))
    eff_dir_8 = np.sign(data['efficiency_momentum'] - data['efficiency_momentum'].shift(8))
    eff_dir_12 = np.sign(data['efficiency_momentum'] - data['efficiency_momentum'].shift(12))
    
    gap_consistency = (gap_dir_5 == gap_dir_8).astype(int) + (gap_dir_5 == gap_dir_12).astype(int)
    eff_consistency = (eff_dir_5 == eff_dir_8).astype(int) + (eff_dir_5 == eff_dir_12).astype(int)
    
    alignment_score = gap_eff_corr * (gap_consistency + eff_consistency) / 4
    
    # Gap-Efficiency Momentum
    gap_eff_momentum = data['gap_momentum'] * data['efficiency_momentum'] * alignment_score
    gap_eff_momentum_weighted = gap_eff_momentum * (1 + data['gap_acceleration'])
    
    # Apply Volatility-Regime Filtering
    # Volatility Measures
    tr1 = data['high'] - data['low']
    tr2 = (data['high'] - data['close'].shift(1)).abs()
    tr3 = (data['low'] - data['close'].shift(1)).abs()
    data['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr_10'] = data['true_range'].rolling(window=10).mean()
    
    # Volatility Regimes
    atr_percentile = data['atr_10'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    high_vol_regime = (atr_percentile > 0.8).astype(float)
    low_vol_regime = (atr_percentile < 0.2).astype(float)
    normal_vol_regime = ((atr_percentile >= 0.2) & (atr_percentile <= 0.8)).astype(float)
    
    # Regime-Adaptive Weighting
    vol_weighted_momentum = (
        high_vol_regime * data['efficiency_momentum'] * 1.5 +
        low_vol_regime * data['gap_momentum'] * 1.2 +
        normal_vol_regime * gap_eff_momentum_weighted
    )
    
    # Integrate Volume Confirmation
    # Volume Dynamics
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_spike'] = data['volume'] / data['volume_5d_avg']
    
    high_low_range_adj = high_low_range.replace(0, np.nan)
    data['volume_efficiency'] = data['volume'] / high_low_range_adj
    data['volume_momentum'] = data['volume_efficiency'] - data['volume_efficiency'].shift(5)
    
    # Volume Filtering
    volume_weight = np.where(
        data['volume_spike'] > 1.5, 1.5,
        np.where(data['volume_spike'] < 0.7, 0.5, 1.0)
    )
    
    # Volume-Weighted Signals
    volume_weighted_signal = vol_weighted_momentum * volume_weight * (1 + data['volume_momentum'])
    
    # Generate Composite Alpha Factor
    # Multi-Timeframe Enhancement
    momentum_5d = volume_weighted_signal.rolling(window=5).mean()
    alignment_8d = alignment_score.rolling(window=8).mean()
    consistency_12d = ((gap_consistency + eff_consistency) / 4).rolling(window=12).mean()
    
    # Final Composite
    composite_alpha = (
        volume_weighted_signal * 0.4 +
        momentum_5d * 0.3 +
        alignment_8d * 0.2 +
        consistency_12d * 0.1
    )
    
    return composite_alpha
