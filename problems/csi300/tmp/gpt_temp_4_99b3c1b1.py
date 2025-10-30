import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['prev_close'] = df['close'].shift(1)
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['gap_magnitude'] = abs(df['gap'])
    df['gap_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['closing_strength'] = (df['close'] - (df['high'] + df['low'])/2) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume calculations
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1).replace(0, np.nan)
    df['volume_5d_avg'] = df['volume'].rolling(5).mean()
    df['volume_concentration'] = df['volume'] / (df['volume'].shift(1) + df['volume'].shift(2) + df['volume'].shift(3)).replace(0, np.nan)
    
    # Gap persistence and momentum
    df['gap_persistence'] = df['gap'].rolling(3).apply(lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / 3 if len(x) == 3 else np.nan)
    df['gap_3d_return'] = df['gap'].rolling(3).sum()
    df['gap_10d_return'] = df['gap'].rolling(10).sum()
    
    # Multi-Timeframe Gap-Volume Divergence
    df['gap_volume_divergence'] = (df['gap_efficiency'] - df['volume_ratio']) * df['gap_persistence']
    df['gap_volume_momentum'] = (df['gap_3d_return'] / df['gap_10d_return'].replace(0, np.nan)) - (df['volume_ratio'].rolling(3).mean() / df['volume_ratio'].rolling(10).mean().replace(0, np.nan))
    df['divergence_acceleration'] = df['gap_volume_divergence'] - df['gap_volume_divergence'].shift(3)
    
    # Gap-Volume Synchronization Quality
    df['gap_volume_alignment'] = np.sign(df['gap_efficiency']) * np.sign(df['volume_ratio'])
    df['gap_volume_consistency'] = 1 - (df['gap_volume_divergence'].rolling(5).std() / abs(df['gap_volume_divergence'].rolling(5).mean()).replace(0, np.nan))
    df['gap_volume_concentration'] = df['gap_efficiency'] * df['volume_concentration']
    
    # Multi-Scale Efficiency Breakout Patterns
    df['prev_5d_high'] = df['high'].shift(1).rolling(5).max()
    df['prev_5d_low'] = df['low'].shift(1).rolling(5).min()
    df['gap_resistance_breakthrough'] = (df['close'] - df['prev_5d_high']) / (df['high'] - df['low']).replace(0, np.nan)
    df['gap_support_breakdown'] = (df['prev_5d_low'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)
    df['volume_direction_sync'] = np.sign(df['close'] - df['open']) * np.sign(df['volume'] - df['volume_5d_avg'])
    df['gap_breakout_validation'] = (df['gap_resistance_breakthrough'] + df['gap_support_breakdown']) * df['volume_direction_sync']
    df['efficiency_momentum'] = df['range_efficiency'] * df['closing_strength'] * df['gap_persistence']
    
    # Volume-Volatility Gap Dynamics
    df['gap_volatility_ratio'] = df['gap_magnitude'] / (df['gap_magnitude'].shift(1) + df['gap_magnitude'].shift(2) + df['gap_magnitude'].shift(3)).replace(0, np.nan) * 3
    df['gap_regime_multiplier'] = np.where(df['gap_volatility_ratio'] > 1.2, 1.5, np.where(df['gap_volatility_ratio'] < 0.8, 0.5, 1.0))
    df['volume_volatility_sync'] = np.sign(df['close'] - df['open']) * np.sign(df['volume'] - df['volume_5d_avg'])
    df['amount_efficiency'] = df['amount'] / (df['volume'] * df['close']).replace(0, np.nan)
    df['volume_weighted_gap_momentum'] = df['gap_efficiency'] * df['volume_ratio']
    
    # Gap-Efficiency Divergence Resolution
    df['gap_efficiency_acceleration'] = (df['gap_efficiency'].rolling(5).mean() - df['gap_efficiency'].rolling(10).mean()) / (df['high'] - df['low']).replace(0, np.nan)
    df['divergence_resolution_strength'] = df['gap_persistence'] * df['gap_volume_momentum'] * df['gap_magnitude']
    df['volume_confirmed_gap_breakout'] = df['gap_breakout_validation'] * df['volume_ratio']
    df['efficiency_enhanced_breakout'] = df['range_efficiency'] * df['closing_strength'] * df['gap_efficiency']
    df['gap_acceleration_confirmation'] = df['gap_efficiency_acceleration'] * df['volume_weighted_gap_momentum']
    
    # Adaptive Gap-Volume Alpha Synthesis
    df['gap_volume_divergence_momentum'] = df['gap_volume_momentum'] * df['divergence_acceleration']
    df['gap_efficiency_enhancement'] = df['gap_efficiency_acceleration'] * df['volume_weighted_gap_momentum']
    df['breakout_validation_composite'] = df['volume_confirmed_gap_breakout'] * df['efficiency_enhanced_breakout']
    
    # Composite Alpha Factors
    df['primary_gap_factor'] = df['gap_volume_divergence_momentum'] * df['gap_efficiency_enhancement']
    df['breakout_confirmation'] = df['breakout_validation_composite'] * df['gap_acceleration_confirmation']
    df['volatility_adjustment'] = df['primary_gap_factor'] * df['gap_regime_multiplier']
    
    # Final alpha calculation
    alpha = df['volatility_adjustment'] * df['divergence_resolution_strength'] * df['amount_efficiency']
    
    # Clean up and return
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    return alpha.fillna(0)
