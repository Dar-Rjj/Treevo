import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Decoupling Framework
    # Price-Volume Decoupling
    data['momentum_divergence'] = (data['close'] / data['close'].shift(1) - 1) - (data['volume'] / data['volume'].rolling(5).mean() - 1)
    data['acceleration_decoupling'] = (data['close'] / data['close'].shift(1) - 1) - (data['close'].shift(1) / data['close'].shift(2) - 1) - (data['volume'] / data['volume'].shift(1) - 1)
    
    # Decoupling persistence
    data['momentum_div_sign'] = np.sign(data['momentum_divergence'])
    data['decoupling_persistence'] = data['momentum_div_sign'].rolling(5).apply(lambda x: np.sum(x == x.shift(1)) if len(x) == 5 else np.nan, raw=False)
    
    # Multi-timeframe Decoupling
    data['short_term_decoupling'] = data['momentum_divergence'] / data['momentum_divergence'].rolling(3).mean() - 1
    data['medium_term_decoupling'] = data['momentum_divergence'] - data['momentum_divergence'].rolling(10).mean()
    data['decoupling_stability'] = data['momentum_divergence'].rolling(20).std() / (data['momentum_divergence'].abs().rolling(20).mean() + 0.001)
    
    # Fractal Regime Detection
    # Volatility Regime Classification
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['vol_regime_mean'] = data['daily_range'].rolling(20).mean()
    data['high_vol_regime'] = data['daily_range'] > data['vol_regime_mean']
    data['low_vol_regime'] = data['daily_range'] < (0.5 * data['vol_regime_mean'])
    data['transition_regime'] = ~(data['high_vol_regime'] | data['low_vol_regime'])
    
    # Volume Regime Classification
    data['volume_mean'] = data['volume'].rolling(20).mean()
    data['volume_std'] = data['volume'].rolling(20).std()
    data['high_volume_regime'] = data['volume'] > (data['volume_mean'] + data['volume_std'])
    data['low_volume_regime'] = data['volume'] < (data['volume_mean'] - data['volume_std'])
    data['normal_volume_regime'] = ~(data['high_volume_regime'] | data['low_volume_regime'])
    
    # Opening Gap Analysis
    data['gap_fill_efficiency'] = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1)).replace(0, np.nan)
    data['gap_direction'] = np.sign(data['open'] / data['close'].shift(1) - 1)
    data['gap_direction_persistence'] = data['gap_direction'].rolling(3).apply(lambda x: np.sum(x == x.shift(1)) if len(x) == 3 else np.nan, raw=False)
    data['gap_volatility_ratio'] = (data['open'] / data['close'].shift(1) - 1).abs() / data['daily_range']
    
    # Gap-Momentum Integration
    data['gap_momentum_alignment'] = (data['open'] / data['close'].shift(1) - 1) * data['momentum_divergence']
    data['gap_acceleration'] = (data['open'] / data['close'].shift(1) - 1) - (data['open'].shift(1) / data['close'].shift(2) - 1)
    data['gap_momentum_decoupling'] = data['gap_acceleration'] - data['momentum_divergence']
    
    # Amount-Based Signal Confirmation
    data['average_trade_size'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['size_momentum'] = data['average_trade_size'] / data['average_trade_size'].rolling(5).mean() - 1
    data['size_volatility'] = data['average_trade_size'].rolling(10).std() / data['average_trade_size'].rolling(10).mean()
    
    # Size-Price Integration
    data['size_momentum_correlation'] = data['size_momentum'] * data['momentum_divergence']
    data['size_gap_interaction'] = data['size_momentum'] * (data['open'] / data['close'].shift(1) - 1)
    data['size_efficiency'] = data['size_momentum'] * data['gap_fill_efficiency']
    
    # Range Breakout Dynamics
    data['true_range_breakout'] = (data['high'] - data['high'].rolling(5).max().shift(1)) - (data['low'].rolling(5).min().shift(1) - data['low'])
    data['breakout_confirmation'] = (data['close'] > data['high'].rolling(5).max().shift(1)).astype(int) - (data['close'] < data['low'].rolling(5).min().shift(1)).astype(int)
    data['breakout_momentum'] = data['true_range_breakout'] / data['true_range_breakout'].abs().rolling(10).mean()
    
    # Breakout-Decoupling Integration
    data['breakout_momentum_alignment'] = data['breakout_confirmation'] * data['momentum_divergence']
    data['range_efficiency'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(10).mean()
    data['breakout_range_ratio'] = data['true_range_breakout'] / data['range_efficiency']
    
    # Multi-scale Fractal Timing
    # Short-term Patterns
    data['momentum_reversal'] = data['momentum_divergence'] * (data['momentum_divergence'].shift(1) < 0)
    data['volume_confirmation'] = (data['volume'] > data['volume'].shift(1)).astype(int) * data['momentum_divergence']
    data['gap_continuation'] = data['gap_direction_persistence'] * data['momentum_divergence']
    
    # Medium-term Patterns
    data['decoupling_trend'] = data['momentum_divergence'].rolling(5).mean()
    data['volume_trend_alignment'] = (data['volume'] / data['volume'].rolling(5).mean() - 1).rolling(10).mean()
    data['gap_efficiency_trend'] = data['gap_fill_efficiency'].rolling(10).mean()
    
    # Long-term Context
    data['regime_persistence'] = data['high_vol_regime'].rolling(20).apply(lambda x: np.sum(x == x.shift(1)) if len(x) == 20 else np.nan, raw=False)
    data['volume_regime_stability'] = data['high_volume_regime'].rolling(20).apply(lambda x: np.sum(x == x.shift(1)) if len(x) == 20 else np.nan, raw=False)
    
    # Regime-Specific Alpha Components
    # High Volatility Regime
    data['volatility_breakout'] = data['true_range_breakout'] * data['range_efficiency']
    data['gap_volatility_premium'] = data['gap_volatility_ratio'] * data['momentum_divergence']
    data['size_volatility_alignment'] = data['size_volatility'] * data['momentum_divergence']
    
    # Low Volatility Regime
    data['accumulation_signal'] = data['momentum_divergence'].rolling(5).sum()
    data['gap_efficiency_premium'] = data['gap_fill_efficiency'] * data['decoupling_persistence']
    data['size_momentum_confirmation'] = data['size_momentum'] * data['accumulation_signal']
    
    # Transition Regime
    data['regime_change_momentum'] = data['momentum_divergence'] * (data['high_vol_regime'] != data['high_vol_regime'].shift(1))
    data['volume_spike_alignment'] = (data['volume'] / data['volume'].rolling(5).mean() - 1) * data['momentum_divergence']
    data['gap_regime_signal'] = data['gap_acceleration'] * data['regime_change_momentum']
    
    # Composite Alpha Generation
    # Core Decoupling Momentum
    data['multi_timeframe_decoupling'] = data['short_term_decoupling'] * data['medium_term_decoupling']
    data['volume_regime_adjusted'] = data['multi_timeframe_decoupling'] * (1 + data['high_volume_regime'])
    data['gap_efficiency_enhanced'] = data['volume_regime_adjusted'] * data['gap_fill_efficiency']
    
    # Breakout Confirmation Layer
    data['breakout_alignment'] = data['gap_efficiency_enhanced'] * data['breakout_momentum_alignment']
    data['range_normalized'] = data['breakout_alignment'] / data['range_efficiency']
    data['size_validated'] = data['range_normalized'] * data['size_efficiency']
    
    # Regime-Adaptive Weighting
    data['volatility_regime_multiplier'] = np.where(data['high_vol_regime'], 1.2, np.where(data['low_vol_regime'], 0.8, 1.0))
    data['volume_regime_confidence'] = np.where(data['high_volume_regime'], 1.1, np.where(data['low_volume_regime'], 0.9, 1.0))
    data['regime_persistence_score'] = (data['regime_persistence'] + data['volume_regime_stability']) / 40
    
    # Final Alpha Factor
    data['regime_weighted_momentum'] = data['size_validated'] * data['volatility_regime_multiplier'] * data['volume_regime_confidence']
    data['fractal_timing'] = data['regime_weighted_momentum'] * data['decoupling_stability']
    data['gap_filtered'] = data['fractal_timing'] * data['gap_direction_persistence']
    data['final_alpha'] = data['gap_filtered'] * data['breakout_confirmation']
    
    return data['final_alpha']
