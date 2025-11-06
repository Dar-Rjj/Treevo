import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Quantum Volatility Regime Detection
    # Multi-Scale Volatility Patterns
    data['short_term_quantum_vol'] = data['close'].pct_change().rolling(window=5).std()
    data['range_quantum_vol'] = (data['high'] - data['low']) / data['close']
    data['volume_adj_vol'] = ((data['high'] - data['low']) / data['close']) * (data['volume'] / data['volume'].shift(1))
    
    # Quantum Efficiency Assessment
    data['quantum_dir_efficiency'] = abs(data['close'] - data['close'].shift(2)) / (
        data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min()
    )
    data['quantum_vol_efficiency'] = data['volume'] / data[['volume', 'volume']].shift(1).min(axis=1)
    
    quantum_eff_5d = data['quantum_dir_efficiency'].rolling(window=5).mean()
    quantum_eff_2d = data['quantum_dir_efficiency'].rolling(window=2).mean()
    data['quantum_eff_acceleration'] = quantum_eff_5d - quantum_eff_2d
    
    # Regime Classification
    high_vol_regime = (data['short_term_quantum_vol'] > 0.02) & (data['range_quantum_vol'] > 0.03)
    low_vol_regime = (data['short_term_quantum_vol'] < 0.01) & (data['range_quantum_vol'] < 0.015)
    normal_vol_regime = ~high_vol_regime & ~low_vol_regime
    
    # Microstructure Momentum Components
    # Price microstructure momentum
    hl_range = data['high'] - data['low']
    data['quantum_dir_bias'] = ((data['close'] - data['open']) / hl_range) * (
        ((data['close'] - data['low']) / hl_range) - ((data['high'] - data['close']) / hl_range)
    )
    data['quantum_level_states'] = ((data['close'] - data['low']) / hl_range) - ((data['high'] - data['close']) / hl_range)
    
    high_diff = abs(data['high'] - data['high'].shift(1))
    low_diff = abs(data['low'] - data['low'].shift(1))
    data['quantum_anchor_convergence'] = (abs(data['close'] - data['close'].shift(1)) / hl_range) - (
        np.minimum(high_diff, low_diff) / hl_range
    )
    
    # Volume microstructure momentum
    up_days = data['close'] > data['open']
    down_days = data['close'] < data['open']
    
    up_volume_5d = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[up_days.loc[x.index].values]) if len(x) == 5 else np.nan, raw=False
    )
    down_volume_5d = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[down_days.loc[x.index].values]) if len(x) == 5 else np.nan, raw=False
    )
    data['quantum_volume_asymmetry'] = up_volume_5d / down_volume_5d
    
    price_change = data['close'] - data['close'].shift(1)
    data['quantum_liquidity_flow'] = (abs(price_change) / data['volume']) * np.sign(price_change) * (data['volume'] / data['volume'].shift(1))
    
    volume_increase = data['volume'] > data['volume'].shift(1)
    data['volume_persistence'] = volume_increase.rolling(window=5).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] and x.iloc[i-1]]), raw=False
    )
    
    # Opening microstructure dynamics
    data['quantum_gap_absorption'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * (
        (data['close'] - data['open']) / hl_range
    )
    data['quantum_auction_imbalance'] = ((data['open'] - data['low']) - (data['high'] - data['open'])) * (data['volume'] / data['volume'].shift(1))
    
    open_low_ratio = (data['open'] - data['low']) / hl_range
    high_open_ratio = (data['high'] - data['open']) / hl_range
    data['opening_quantum_state'] = np.where((open_low_ratio > 0.7) | (high_open_ratio > 0.7), 1.5, 1.0)
    
    # Regime-Adaptive Signal Processing
    # High quantum volatility regime
    high_vol_core = -1 * data['quantum_dir_bias'] / (1 + data['range_quantum_vol']) * (
        1 + data['quantum_volume_asymmetry'] * abs(data['quantum_dir_bias'])
    )
    high_vol_enhanced = high_vol_core * data['quantum_anchor_convergence'] * data['opening_quantum_state']
    
    # Low quantum volatility regime
    price_dir_1 = np.sign(data['close'] - data['close'].shift(1))
    price_dir_2 = np.sign(data['close'].shift(1) - data['close'].shift(2))
    low_vol_core = data['quantum_dir_bias'] * price_dir_1 * price_dir_2 * (1 + data['quantum_liquidity_flow'])
    low_vol_enhanced = low_vol_core * data['quantum_level_states'] * data['volume_persistence']
    
    # Normal quantum volatility regime
    momentum_3d = data['close'] / data['close'].shift(3) - 1
    momentum_8d = data['close'] / data['close'].shift(8) - 1
    normal_vol_core = 0.6 * momentum_3d + 0.4 * momentum_8d * data['quantum_eff_acceleration']
    normal_vol_enhanced = normal_vol_core * data['quantum_volume_asymmetry'] * data['quantum_gap_absorption']
    
    # Select regime-adaptive signal
    regime_signal = np.where(high_vol_regime, high_vol_enhanced,
                           np.where(low_vol_regime, low_vol_enhanced, normal_vol_enhanced))
    
    # Convergence Validation Framework
    # Multi-timeframe alignment
    ultra_short_conv = data['quantum_eff_acceleration'] * data['quantum_dir_bias']
    short_term_conv = data['quantum_dir_efficiency'].rolling(window=5).mean() * data['quantum_level_states']
    price_momentum_10d = data['close'].pct_change(periods=10)
    medium_term_align = price_momentum_10d * data['quantum_volume_asymmetry']
    
    # Volume confirmation
    vol_momentum_conv = (data['volume'] / data['volume'].shift(5)) * data['quantum_liquidity_flow']
    vol_pattern_align = data['volume_persistence'] * data['quantum_volume_asymmetry']
    
    # Convergence strength assessment
    timeframe_positive = (ultra_short_conv > 0).astype(int) + (short_term_conv > 0).astype(int) + (medium_term_align > 0).astype(int)
    vol_confirmation = vol_momentum_conv
    
    strong_conv = (timeframe_positive == 3) & (vol_confirmation > 1)
    moderate_conv = (timeframe_positive == 2) & (vol_confirmation > 0.5)
    weak_conv = (timeframe_positive == 1) | (vol_confirmation < 0.5)
    
    # Apply convergence multiplier
    convergence_multiplier = np.where(strong_conv, 1.5,
                                    np.where(moderate_conv, 1.2, 0.8))
    
    # Final Alpha Synthesis with risk adjustment
    alpha_factor = (regime_signal * convergence_multiplier) / (1 + data['short_term_quantum_vol'])
    
    return pd.Series(alpha_factor, index=data.index)
