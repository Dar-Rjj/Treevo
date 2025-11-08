import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Quantum-Volatility Regime Entanglement with Intraday Momentum Fractals
    """
    df = df.copy()
    
    # Quantum Volatility Regime States
    # Volatility calculations
    df['returns'] = df['close'].pct_change()
    df['vol_20d'] = df['returns'].rolling(window=20).std()
    df['vol_60d_median'] = df['vol_20d'].rolling(window=60).median()
    
    # Volume probability and amplitude
    df['volume_20d_mean'] = df['volume'].rolling(window=20).mean()
    df['volume_20d_std'] = df['volume'].rolling(window=20).std()
    df['volume_probability'] = (df['volume'] - df['volume_20d_mean']) / df['volume_20d_std']
    df['volume_amplitude'] = df['volume'] / df['volume_20d_mean']
    
    # Quantum volatility classification
    df['quantum_high_vol'] = ((df['vol_20d'] > df['vol_60d_median']) & 
                              (df['volume_probability'] > 0)).astype(int)
    df['quantum_low_vol'] = ((df['vol_20d'] < df['vol_60d_median']) & 
                             (df['volume_amplitude'] < 1.2)).astype(int)
    df['mixed_vol_state'] = ((df['vol_20d'] > df['vol_60d_median']) != 
                            (df['volume_probability'] > 0)).astype(int)
    
    # Multi-scale volatility clustering
    df['vol_5d'] = df['returns'].rolling(window=5).std()
    df['vol_10d'] = df['returns'].rolling(window=10).std()
    df['vol_clustering'] = (df['vol_5d'] * df['vol_10d'] * df['vol_20d']) ** (1/3)
    
    # Intraday Momentum Quantum Fractals
    # Morning momentum (assuming first hour data available)
    df['intraday_range'] = df['high'] - df['low']
    df['morning_strength'] = (df['high'].rolling(window=5).apply(lambda x: x[-1] - x[0]) / 
                             df['intraday_range'].rolling(window=5).mean())
    
    # Intraday efficiency
    df['intraday_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['efficiency_5d'] = df['intraday_efficiency'].rolling(window=5).mean()
    df['efficiency_10d'] = df['intraday_efficiency'].rolling(window=10).mean()
    
    # Gap interactions
    df['prev_close'] = df['close'].shift(1)
    df['gap_size'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['gap_continuation'] = ((df['close'] - df['open']) / 
                             (abs(df['open'] - df['prev_close'])).replace(0, np.nan))
    
    # Liquidity-Volume Quantum Asymmetry
    # Volume exhaustion
    df['volume_3d_derivative'] = df['volume'].diff(3) / df['volume'].shift(3)
    df['volume_range_ratio'] = df['volume'] / df['intraday_range']
    df['volume_exhaustion'] = (df['volume_3d_derivative'] < 0) & (df['volume_range_ratio'] > df['volume_range_ratio'].rolling(20).quantile(0.8))
    
    # Buying pressure accumulation
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['weighted_return'] = df['daily_return'] * df['volume_amplitude']
    df['buying_pressure'] = df['weighted_return'].rolling(window=5).sum()
    
    # Quantum Efficiency Regime Switching
    # Multi-scale efficiency comparison
    df['efficiency_regime'] = np.where(df['intraday_efficiency'] > df['efficiency_5d'], 1, 
                                      np.where(df['intraday_efficiency'] > df['efficiency_10d'], 0, -1))
    
    # Market state detection
    df['trend_strength'] = df['close'].rolling(window=10).apply(lambda x: (x[-1] - x[0]) / np.std(x))
    df['market_state'] = np.where(abs(df['trend_strength']) > 0.5, 1, 0)  # 1: trending, 0: mean-reverting
    
    # Multi-Scale Quantum-Fractal Alignment
    # Timeframe consistency
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_20d'] = df['close'].pct_change(20)
    df['momentum_alignment'] = (np.sign(df['momentum_5d']) + np.sign(df['momentum_10d']) + np.sign(df['momentum_20d'])) / 3
    
    # Volume-intensity relationships
    df['volume_intensity'] = df['volume'] / df['volume_20d_mean']
    df['volatility_intensity'] = df['vol_20d'] / df['vol_60d_median']
    df['volume_volatility_entanglement'] = df['volume_intensity'] * df['volatility_intensity']
    
    # Quantum-Fractal Alpha Generation
    # High volatility quantum momentum component
    high_vol_momentum = (df['morning_strength'] * df['quantum_high_vol'] * 
                        df['volume_probability'] * df['intraday_efficiency'])
    
    # Low volatility quantum efficiency component
    low_vol_efficiency = (df['efficiency_regime'] * df['quantum_low_vol'] * 
                         df['volume_amplitude'] * df['buying_pressure'])
    
    # Regime transition signals
    vol_regime_change = df['quantum_high_vol'].diff().abs() + df['quantum_low_vol'].diff().abs()
    transition_signals = (vol_regime_change * df['gap_continuation'] * 
                         df['momentum_alignment'] * df['volume_exhaustion'])
    
    # Multi-scale fractal confirmation
    fractal_confirmation = (df['momentum_alignment'] * df['volume_volatility_entanglement'] * 
                          df['market_state'] * df['efficiency_regime'])
    
    # Final alpha factor combining all quantum-fractal components
    alpha = (high_vol_momentum.fillna(0) * 0.4 + 
             low_vol_efficiency.fillna(0) * 0.3 + 
             transition_signals.fillna(0) * 0.2 + 
             fractal_confirmation.fillna(0) * 0.1)
    
    # Risk-aware normalization
    alpha_vol = alpha.rolling(window=20).std()
    normalized_alpha = alpha / alpha_vol.replace(0, np.nan)
    
    return normalized_alpha
