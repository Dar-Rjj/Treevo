import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Adjusted Efficiency Framework
    df['short_term_momentum_efficiency'] = np.abs(df['close'] - df['open']) / np.abs(df['close'] / df['close'].shift(1) - 1).replace(0, np.nan)
    df['medium_term_momentum_absorption'] = np.abs(df['close'] - df['open']) / np.abs(df['close'] / df['close'].shift(3) - 1).replace(0, np.nan)
    df['efficiency_divergence'] = df['short_term_momentum_efficiency'] - df['medium_term_momentum_absorption']
    
    df['intraday_efficiency'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['volatility_scaled_efficiency'] = df['intraday_efficiency'] / (df['high'] - df['low']).replace(0, np.nan)
    df['efficiency_momentum'] = df['intraday_efficiency'] / df['intraday_efficiency'].rolling(5).mean() - 1
    
    df['volatility_adjusted_reversal'] = -(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['reversal_persistence'] = df['volatility_adjusted_reversal'].rolling(3).apply(lambda x: (x > 0).sum() - (x < 0).sum())
    df['multi_scale_reversal'] = df['volatility_adjusted_reversal'] * df['efficiency_divergence']
    
    # Volume-Efficiency Alignment System
    df['directional_volume'] = np.sign(df['close'] - df['close'].shift(1)) * df['volume']
    df['volume_pressure'] = df['volume'] / df['volume'].rolling(5).mean()
    df['volume_acceleration'] = df['volume_pressure'] / df['volume_pressure'].rolling(3).mean() - 1
    
    df['amount_intensity'] = df['amount'] / df['amount'].rolling(5).mean()
    df['amount_volume_ratio'] = df['amount_intensity'] / df['volume_pressure'].replace(0, np.nan)
    df['volume_efficiency_alignment'] = df['directional_volume'] * df['intraday_efficiency']
    
    df['range_breakout'] = (df['close'] - df['high'].shift(1).rolling(5).max()) / (df['high'] - df['low']).replace(0, np.nan)
    df['volume_weighted_breakout'] = df['range_breakout'] * df['volume_pressure']
    df['efficiency_confirmed_breakout'] = df['volume_weighted_breakout'] * df['amount_volume_ratio']
    
    # Gap-Persistence Momentum Framework
    df['opening_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_persistence'] = df['opening_gap'] * (df['close'] - df['open']) / df['close'].shift(1)
    df['volatility_adjusted_gap'] = df['opening_gap'] / (df['high'] - df['low']).replace(0, np.nan)
    
    df['intraday_momentum'] = (df['close'] - df['open']) / df['open']
    df['gap_momentum_alignment'] = df['gap_persistence'] * df['intraday_momentum']
    df['volume_confirmed_gap'] = df['volatility_adjusted_gap'] * df['volume_pressure']
    
    df['gap_momentum_persistence'] = df['gap_momentum_alignment'].rolling(3).apply(lambda x: (x > 0).sum() - (x < 0).sum())
    df['volume_weighted_persistence'] = df['gap_momentum_persistence'] * df['volume_acceleration']
    df['efficiency_enhanced_gap'] = df['gap_persistence'] * df['intraday_efficiency']
    
    # Regime-Adaptive Structure
    df['volatility_ratio'] = (df['high'] - df['low']).rolling(3).mean() / (df['high'] - df['low']).rolling(8).mean()
    returns = df['close'] / df['close'].shift(1) - 1
    df['volatility_adjusted_momentum'] = returns / returns.rolling(10).std()
    df['momentum_acceleration'] = (df['close'] / df['close'].shift(2) - 1) - (df['close'] / df['close'].shift(5) - 1)
    
    df['efficiency_persistence'] = df['intraday_efficiency'].rolling(5).apply(lambda x: (x > 0.6).sum() - (x < 0.4).sum())
    df['efficiency_volatility'] = df['intraday_efficiency'].rolling(5).std()
    vol_std = returns.rolling(10).std()
    df['regime_weight'] = 1 / (1 + np.abs(vol_std - vol_std.rolling(20).mean()))
    
    df['volatility_scaled_efficiency_regime'] = df['efficiency_divergence'] * df['volatility_ratio']
    df['regime_weighted_momentum'] = df['momentum_acceleration'] * df['regime_weight']
    df['volume_adaptive_reversal'] = df['multi_scale_reversal'] * df['volume_pressure']
    
    # Composite Alpha Synthesis
    df['efficiency_momentum_alignment'] = df['efficiency_divergence'] * df['momentum_acceleration']
    df['volume_confirmed_breakout_composite'] = df['efficiency_confirmed_breakout'] * df['volume_efficiency_alignment']
    df['gap_persistence_momentum'] = df['efficiency_enhanced_gap'] * df['volume_weighted_persistence']
    
    df['volatility_adjusted_signals'] = df['volatility_scaled_efficiency_regime'] * df['regime_weighted_momentum']
    df['volume_regime_alignment'] = df['volume_adaptive_reversal'] * df['efficiency_persistence']
    df['multi_timeframe_confirmation'] = df['gap_momentum_alignment'] * df['volume_confirmed_breakout_composite']
    
    # Final Alpha Construction
    df['primary_alpha'] = df['efficiency_momentum_alignment'] * df['volume_confirmed_breakout_composite']
    df['confirmation_alpha'] = df['gap_persistence_momentum'] * df['volatility_adjusted_signals']
    df['enhanced_alpha'] = df['primary_alpha'] * df['confirmation_alpha'] * df['volume_regime_alignment']
    
    return df['enhanced_alpha']
