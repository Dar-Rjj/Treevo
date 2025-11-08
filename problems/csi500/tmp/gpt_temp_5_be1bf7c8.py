import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Momentum Acceleration Factor
    Combines multi-period momentum acceleration with volume confirmation,
    gap analysis, and liquidity assessment for predictive factor generation.
    """
    data = df.copy()
    
    # Multi-Period Momentum Acceleration
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    data['mom_accel'] = data['mom_3d'] - data['mom_3d'].shift(3)
    data['mom_divergence'] = data['mom_3d'] - data['mom_10d']
    
    # Fractal Amplitude Integration
    data['amplitude'] = (data['high'] - data['low']) / data['close']
    data['amplitude_5d'] = data['amplitude'].rolling(window=5).mean()
    data['amplitude_10d'] = data['amplitude'].rolling(window=10).mean()
    data['amplitude_mom'] = data['amplitude_5d'] / data['amplitude_10d'].shift(5) - 1
    
    # Amplitude clustering detection
    data['amplitude_std_5d'] = data['amplitude'].rolling(window=5).std()
    data['amplitude_clustering'] = data['amplitude_std_5d'] / data['amplitude_std_5d'].rolling(window=10).mean()
    
    # Volume Momentum and Persistence
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_mom'] = (data['volume'] / data['volume_ma_5d'].shift(5)) - 1
    
    # Volume persistence (consecutive above average days)
    volume_above_avg = (data['volume'] > data['volume_ma_5d']).astype(int)
    data['volume_persistence'] = volume_above_avg.rolling(window=5).sum()
    
    # Volume-Amplitude Alignment
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                   np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                             abs(data['low'] - data['close'].shift(1))))
    data['volume_amplitude_corr'] = data['volume'].rolling(window=10).corr(data['true_range'])
    data['volume_amplitude_divergence'] = (data['volume_mom'] - data['amplitude_mom']).abs()
    
    # Gap-Fractal Momentum Integration
    data['gap_size'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_relative_amplitude'] = abs(data['gap_size']) / data['amplitude_5d']
    data['gap_persistence'] = (np.sign(data['gap_size']) == np.sign(data['gap_size'].shift(1))).astype(int)
    data['gap_filling'] = (abs(data['close'] - data['open']) / abs(data['gap_size'] * data['close'].shift(1))).fillna(0)
    
    # Gap-Momentum Acceleration Synthesis
    data['gap_momentum_alignment'] = np.sign(data['gap_size']) * np.sign(data['mom_accel'])
    data['gap_volume_confirmation'] = data['gap_momentum_alignment'] * data['volume_persistence']
    
    # Liquidity-Enhanced Acceleration Signals
    data['trade_size'] = data['amount'] / data['volume']
    data['trade_size_ma'] = data['trade_size'].rolling(window=5).mean()
    data['large_trade_ratio'] = (data['trade_size'] > data['trade_size_ma']).astype(int)
    data['liquidity_efficiency'] = data['true_range'] / data['volume']
    
    # Liquidity-Momentum Alignment
    data['liquidity_trend'] = data['trade_size'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    data['liquidity_momentum_divergence'] = (data['liquidity_trend'] * data['mom_accel']).abs()
    
    # Adaptive Composite Factor Generation
    # Momentum Acceleration Weighting
    data['volume_confirmation_strength'] = data['volume_persistence'] * data['volume_amplitude_corr'].abs()
    data['momentum_weighted'] = data['mom_accel'] * data['volume_confirmation_strength']
    
    # Fractal Pattern Integration
    data['amplitude_stability'] = 1 / (1 + data['amplitude_clustering'])
    data['fractal_momentum'] = data['momentum_weighted'] * data['amplitude_stability']
    
    # Gap persistence weighting
    data['gap_weight'] = data['gap_persistence'] * (1 - data['gap_filling'])
    data['gap_adjusted_momentum'] = data['fractal_momentum'] * (1 + data['gap_weight'] * np.sign(data['gap_momentum_alignment']))
    
    # Persistence-Enhanced Signal Construction
    data['direction_persistence'] = (np.sign(data['mom_accel']) == np.sign(data['mom_accel'].shift(1))).astype(int)
    data['persistence_enhanced'] = data['gap_adjusted_momentum'] * (1 + 0.2 * data['direction_persistence'])
    
    # Volume trend confirmation
    data['volume_trend'] = data['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    data['volume_confirmed'] = data['persistence_enhanced'] * np.sign(data['volume_trend'])
    
    # Liquidity stability weighting
    data['liquidity_stability'] = 1 / (1 + data['liquidity_efficiency'].rolling(window=5).std())
    data['liquidity_weighted'] = data['volume_confirmed'] * data['liquidity_stability']
    
    # Acceleration consistency
    data['accel_consistency'] = data['mom_accel'].rolling(window=5).std()
    data['consistency_adjusted'] = data['liquidity_weighted'] / (1 + data['accel_consistency'])
    
    # Final Predictive Factor
    data['fractal_momentum_accel_factor'] = (
        data['consistency_adjusted'] * 
        data['amplitude_stability'] * 
        (1 + 0.1 * data['gap_volume_confirmation'])
    )
    
    # Clean up and return
    factor = data['fractal_momentum_accel_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
