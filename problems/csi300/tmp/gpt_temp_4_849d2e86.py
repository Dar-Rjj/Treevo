import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price differences and ranges
    df['close_diff'] = df['close'].diff()
    df['high_low_range'] = df['high'] - df['low']
    df['prev_high_low_range'] = df['high_low_range'].shift(1)
    df['volume_diff'] = df['volume'].diff()
    
    # Price Fractal Dimension Analysis
    df['fractal_range_ratio'] = df['high_low_range'] / df['prev_high_low_range']
    
    # Price Compression
    df['range_ma_5'] = df['high_low_range'].rolling(window=5, min_periods=1).mean().shift(1)
    df['price_compression'] = df['high_low_range'] / df['range_ma_5']
    
    # Opening Fractal Gap
    df['opening_fractal_gap'] = abs(df['open'] - df['close'].shift(1)) / df['prev_high_low_range']
    
    # Fractal Momentum
    df['close_diff_prev'] = df['close_diff'].shift(1)
    df['fractal_momentum'] = df['close_diff'] / df['close_diff_prev']
    
    # Pattern Persistence
    df['price_sign'] = np.sign(df['close_diff'])
    df['pattern_persistence'] = df['price_sign'].rolling(window=5, min_periods=1).apply(
        lambda x: sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    # Fractal Volatility
    df['volatility_short'] = df['close'].rolling(window=5, min_periods=1).std()
    df['volatility_long'] = df['close'].rolling(window=5, min_periods=1).std().shift(5)
    df['fractal_volatility'] = df['volatility_short'] / df['volatility_long']
    
    # Behavioral Volume Dynamics
    df['volume_scaling'] = df['volume'] / df['volume'].shift(1)
    
    # Volume Persistence Score
    df['volume_gt_prev'] = (df['volume'] > df['volume'].shift(1)).astype(int)
    df['volume_persistence_score'] = df['volume_gt_prev'].rolling(window=5, min_periods=1).sum()
    
    # Volume Fractal Dimension
    df['volume_fractal_dim'] = np.log(df['volume']) / np.log(df['high_low_range'])
    
    # Herding Volume
    df['volume_ma_3'] = df['volume'].rolling(window=3, min_periods=1).mean().shift(1)
    df['herding_volume'] = df['volume'] / df['volume_ma_3']
    
    # Volume Momentum Divergence
    df['volume_momentum_divergence'] = np.sign(df['volume_diff']) * np.sign(df['close_diff'])
    
    # Volume Exhaustion
    df['volume_exhaustion'] = np.where(df['volume'] > 2 * df['volume'].shift(1), -1, 1)
    
    # Market Microstructure Regimes
    df['efficiency_score'] = abs(df['close'] - (df['high'] + df['low'])/2) / df['high_low_range']
    df['opening_efficiency'] = abs(df['open'] - df['close'].shift(1)) / df['high_low_range']
    df['total_microstructure_efficiency'] = df['efficiency_score'] + df['opening_efficiency']
    
    # Market Regime Classification
    df['high_vol_regime'] = ((df['fractal_range_ratio'] > 1.5) & (df['volume_scaling'] > 1.3)).astype(int)
    df['low_vol_regime'] = ((df['fractal_range_ratio'] < 0.7) & (df['volume_scaling'] < 0.8)).astype(int)
    df['transition_regime'] = ((df['high_vol_regime'] == 0) & (df['low_vol_regime'] == 0)).astype(int)
    
    # Momentum Fractal Integration
    df['short_term_momentum'] = df['close'] / df['close'].shift(2) - 1
    df['medium_term_momentum'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_fractal_ratio'] = df['short_term_momentum'] / df['medium_term_momentum']
    
    # Price Level Dynamics
    df['support_level_strength'] = (df['close'] - df['low']) / df['high_low_range']
    df['resistance_level_strength'] = (df['high'] - df['close']) / df['high_low_range']
    df['level_compression'] = np.where((df['support_level_strength'] < 0.25) & 
                                     (df['resistance_level_strength'] < 0.25), 1.4, 1.0)
    
    # Volume-Price Fractal Alignment
    df['volume_confirmed_momentum'] = df['medium_term_momentum'] * df['volume_persistence_score']
    df['fractal_volume_alignment'] = df['volume_fractal_dim'] * df['momentum_fractal_ratio']
    df['behavioral_confirmation'] = df['volume_confirmed_momentum'] * df['herding_volume']
    
    # Microstructure Flow Analysis
    df['intraday_flow'] = (df['close'] - df['open']) / df['high_low_range']
    df['volume_flow_direction'] = np.sign(df['volume_diff']) * np.sign(df['close'] - df['open'])
    
    # Flow Consistency
    df['flow_consistency'] = df['volume_flow_direction'].rolling(window=3, min_periods=1).apply(
        lambda x: sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    # Fractal Regime Synthesis
    df['high_vol_multiplier'] = np.where(df['high_vol_regime'] == 1, 1.3, 1.0)
    df['low_vol_multiplier'] = np.where(df['low_vol_regime'] == 1, 0.8, 1.0)
    df['transition_multiplier'] = np.where(df['transition_regime'] == 1, 1.1, 1.0)
    
    df['regime_multiplier'] = (df['high_vol_multiplier'] * df['high_vol_regime'] + 
                              df['low_vol_multiplier'] * df['low_vol_regime'] + 
                              df['transition_multiplier'] * df['transition_regime'])
    
    # Behavioral Pattern Integration
    df['pattern_strength'] = df['pattern_persistence'] * df['volume_persistence_score']
    df['fractal_consistency'] = np.where(df['pattern_strength'] > 3, 1.2, 1.0)
    
    # Advanced Microstructure Integration
    df['efficiency_weighted_momentum'] = df['medium_term_momentum'] * df['total_microstructure_efficiency']
    df['volume_efficiency_alignment'] = df['efficiency_weighted_momentum'] * df['volume_fractal_dim']
    df['level_enhanced_signal'] = df['volume_efficiency_alignment'] * df['level_compression']
    
    # Flow-Based Refinement
    df['flow_confirmed_signal'] = df['level_enhanced_signal'] * df['flow_consistency']
    df['volume_divergence_adjusted'] = df['flow_confirmed_signal'] * df['volume_momentum_divergence']
    df['behavioral_pattern_filtered'] = df['volume_divergence_adjusted'] * df['fractal_consistency']
    
    # Final Fractal Alpha Output
    df['base_signal'] = df['behavioral_pattern_filtered'] * df['regime_multiplier']
    df['volume_behavioral_adjusted'] = df['base_signal'] * df['behavioral_confirmation']
    df['fractal_momentum_integrated'] = df['volume_behavioral_adjusted'] * df['momentum_fractal_ratio']
    
    # Risk-Adjusted Refinement
    df['volatility_scaled'] = df['fractal_momentum_integrated'] * df['price_compression']
    df['volume_exhaustion_controlled'] = df['volatility_scaled'] * df['volume_exhaustion']
    df['pattern_strength_filtered'] = df['volume_exhaustion_controlled'] * df['fractal_consistency']
    
    # Final Alpha Generation
    df['directional_bias'] = df['pattern_strength_filtered'] * df['volume_flow_direction']
    df['regime_optimized'] = df['directional_bias'] * df['regime_multiplier']
    df['final_alpha'] = df['regime_optimized'] * df['total_microstructure_efficiency']
    
    # Fill NaN values and return
    result = df['final_alpha'].fillna(0)
    
    return result
