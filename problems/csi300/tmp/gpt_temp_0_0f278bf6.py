import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Framework
    # Price Momentum Components
    df['intraday_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['gap_momentum'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['price_momentum_divergence'] = df['intraday_momentum'] - df['gap_momentum']
    
    # Volume Distribution Analysis
    df['volume_concentration'] = df['volume'] / (df['volume'] + df['volume'].shift(1) + df['volume'].shift(2))
    df['volume_momentum'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_price_divergence'] = df['volume_momentum'] * df['price_momentum_divergence']
    
    # Divergence Strength Assessment
    df['abs_divergence'] = np.abs(df['volume_price_divergence'])
    df['directional_consistency'] = np.sign(df['volume_price_divergence']) * np.sign(df['price_momentum_divergence'])
    df['divergence_intensity'] = df['abs_divergence'] * df['directional_consistency']
    
    # Regime-Switching Efficiency Metrics
    # Market Regime Identification
    df['volatility_clustering'] = (df['high'] - df['low']) / (df['high'].shift(2) - df['low'].shift(2)).replace(0, np.nan)
    df['trend_persistence'] = (df['close'] - df['close'].shift(5)) / (df['high'].shift(5) - df['low'].shift(5)).replace(0, np.nan)
    df['regime_switch_signal'] = df['volatility_clustering'] * df['trend_persistence']
    
    # Efficiency Under Different Regimes
    df['high_vol_efficiency'] = np.abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    df['low_vol_efficiency'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['adaptive_efficiency'] = np.where(df['regime_switch_signal'] > 0, df['high_vol_efficiency'], df['low_vol_efficiency'])
    
    # Regime Transition Dynamics
    df['regime_sign'] = np.sign(df['regime_switch_signal'])
    df['regime_duration'] = df['regime_sign'].rolling(window=5, min_periods=1).apply(
        lambda x: (x == x.iloc[0]).sum() if len(x) > 0 else 1, raw=False
    )
    df['regime_strength'] = np.abs(df['regime_switch_signal']) / (1 + df['regime_duration'])
    df['transition_efficiency'] = df['adaptive_efficiency'] * df['regime_strength']
    
    # Momentum Quality Assessment
    # Price Quality Indicators
    df['close_position_quality'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['open_close_consistency'] = np.sign(df['open'] - df['close'].shift(1)) * np.sign(df['close'] - df['open'])
    df['price_quality_score'] = df['close_position_quality'] * df['open_close_consistency']
    
    # Volume Quality Metrics
    df['volume_spike_detection'] = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    df['volume_sustainability'] = df['volume'] / df['volume'].shift(1)
    df['volume_quality_score'] = df['volume_spike_detection'] * df['volume_sustainability']
    
    # Combined Quality Assessment
    df['momentum_quality'] = df['price_quality_score'] * df['volume_quality_score']
    df['quality_weighted_divergence'] = df['divergence_intensity'] * df['momentum_quality']
    df['regime_enhanced_quality'] = df['quality_weighted_divergence'] * df['transition_efficiency']
    
    # Breakout Confirmation System
    # Multi-timeframe Breakout Signals
    df['short_term_breakout'] = (df['high'] - df['high'].shift(3)) / (df['high'].shift(3) - df['low'].shift(3)).replace(0, np.nan)
    df['medium_term_breakout'] = (df['close'] - df['close'].shift(8)) / df['close'].rolling(window=8, min_periods=1).std().shift(1)
    df['breakout_convergence'] = df['short_term_breakout'] * df['medium_term_breakout']
    
    # Volume Confirmation
    df['breakout_volume_ratio'] = df['volume'] / df['volume'].shift(3)
    df['volume_breakout_signal'] = df['breakout_volume_ratio'] * df['breakout_convergence']
    df['confirmed_breakout'] = df['volume_breakout_signal'] * np.sign(df['breakout_convergence'])
    
    # Efficiency-Confirmed Breakout
    df['breakout_efficiency'] = df['confirmed_breakout'] * df['adaptive_efficiency']
    df['quality_enhanced_breakout'] = df['breakout_efficiency'] * df['momentum_quality']
    df['final_breakout_signal'] = df['quality_enhanced_breakout'] * df['regime_strength']
    
    # Alpha Factor Integration
    df['core_divergence_component'] = df['regime_enhanced_quality'] * df['divergence_intensity']
    df['breakout_confirmation'] = df['core_divergence_component'] * df['final_breakout_signal']
    df['efficiency_weighting'] = df['breakout_confirmation'] * df['transition_efficiency']
    df['final_alpha_factor'] = df['efficiency_weighting'] * df['momentum_quality']
    
    return df['final_alpha_factor']
