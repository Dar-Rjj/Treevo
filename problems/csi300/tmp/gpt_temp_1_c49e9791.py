import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic components
    df['upper_rejection'] = df['high'] - df['close']
    df['lower_rejection'] = df['close'] - df['low']
    df['range'] = df['high'] - df['low']
    df['efficiency'] = abs(df['close'] - df['open']) / df['range']
    df['vwap'] = (df['high'] + df['low']) / 2
    df['trade_size'] = df['amount'] / df['volume']
    
    # Directional Rejection Pressure
    df['directional_rejection_pressure'] = (df['upper_rejection'] - df['lower_rejection']) * np.sign(df['close'] - df['close'].shift(1))
    
    # Volatility-Weighted Rejection
    df['volatility_weighted_rejection'] = ((df['upper_rejection'] - df['lower_rejection']) / df['range']) * (df['range'] / df['range'].shift(1))
    
    # Volume-Confirmed Rejection
    df['volume_confirmed_rejection'] = (df['upper_rejection'] - df['lower_rejection']) * (df['volume'] / df['volume'].shift(1))
    
    # Short-Term Rejection Momentum
    df['close_rolling_max_4'] = df['close'].rolling(window=4, min_periods=1).max()
    df['close_rolling_min_4'] = df['close'].rolling(window=4, min_periods=1).min()
    df['short_term_rejection_momentum'] = ((df['high'] - df['close_rolling_max_4']) - (df['close_rolling_min_4'] - df['low'])) / df['range']
    
    # Medium-Term Rejection Acceleration
    df['close_rolling_max_14'] = df['close'].rolling(window=14, min_periods=1).max()
    df['close_rolling_min_14'] = df['close'].rolling(window=14, min_periods=1).min()
    df['medium_term_rejection_acceleration'] = ((df['high'] - df['close_rolling_max_14']) - (df['close_rolling_min_14'] - df['low'])) / df['range'] - df['short_term_rejection_momentum']
    
    # Multi-Scale Rejection Alignment
    df['multi_scale_rejection_alignment'] = df['short_term_rejection_momentum'] * df['medium_term_rejection_acceleration']
    
    # Rejection Efficiency
    df['rejection_efficiency'] = ((df['upper_rejection'] - df['lower_rejection']) / df['range']) * df['efficiency']
    
    # Volume-Rejection Scaling
    df['volume_rejection_scaling'] = ((df['upper_rejection'] - df['lower_rejection']) * df['volume']) / ((df['upper_rejection'].shift(1) - df['lower_rejection'].shift(1)) * df['volume'].shift(1))
    
    # Liquidity Absorption Rejection
    df['liquidity_absorption_rejection'] = ((df['close'] - df['vwap']) / df['range']) * (df['upper_rejection'] - df['lower_rejection'])
    
    # High Volatility Rejection
    df['high_volatility_rejection'] = ((df['upper_rejection'] - df['lower_rejection']) / df['range']) * (df['range'] / df['range'].shift(5))
    
    # Low Volatility Rejection
    df['low_volatility_rejection'] = ((df['upper_rejection'] - df['lower_rejection']) / df['range']) * (df['range'].shift(5) / df['range'])
    
    # Volatility-Rejection Persistence
    rejection_sign = np.sign(df['upper_rejection'] - df['lower_rejection'])
    df['volatility_rejection_persistence'] = rejection_sign.rolling(window=5).apply(lambda x: sum(x == x.shift(1)) / 5, raw=False)
    
    # Volume Surge Rejection
    df['volume_ma_4'] = df['volume'].rolling(window=4).mean()
    df['volume_surge_rejection'] = (df['upper_rejection'] - df['lower_rejection']) * (df['volume'] / df['volume_ma_4'])
    
    # Volume Drought Rejection
    df['volume_drought_rejection'] = (df['upper_rejection'] - df['lower_rejection']) * (df['volume_ma_4'] / df['volume'])
    
    # Volume-Rejection Regime
    df['volume_rejection_regime'] = np.sign(df['volume'] - df['volume'].shift(1)) * np.sign(df['upper_rejection'] - df['lower_rejection'])
    
    # Institutional Rejection
    df['trade_size_ma_4'] = df['trade_size'].rolling(window=4).mean()
    df['institutional_rejection'] = (df['upper_rejection'] - df['lower_rejection']) * (df['trade_size'] / df['trade_size_ma_4'])
    
    # Trade Size Impact Rejection
    df['trade_size_impact_rejection'] = (df['trade_size'] * (df['upper_rejection'] - df['lower_rejection'])) / df['range']
    
    # Size-Regime Alignment
    df['size_regime_alignment'] = np.sign(df['trade_size'] - df['trade_size'].shift(1)) * np.sign(df['upper_rejection'] - df['lower_rejection'])
    
    # Fractal Gap Momentum
    df['fractal_gap_momentum'] = ((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) - df['close'].shift(2))) * (df['volume'] / df['volume'].shift(1))
    
    # Gap Rejection Efficiency
    df['gap_rejection_efficiency'] = (abs(df['open'] - df['close'].shift(1)) / df['range']) * (df['upper_rejection'] - df['lower_rejection'])
    
    # Multi-Scale Gap Absorption
    df['multi_scale_gap_absorption'] = df['fractal_gap_momentum'] * df['gap_rejection_efficiency']
    
    # Opening Efficiency Momentum
    df['opening_efficiency_momentum'] = (df['high'] - df['open']) - (df['open'] - df['low']) * df['efficiency']
    
    # Closing Efficiency Fractal
    df['closing_efficiency_fractal'] = ((df['close'] - df['vwap']) / df['range']) * (df['volume'] / df['volume'].shift(1))
    
    # Session Efficiency Alignment
    df['session_efficiency_alignment'] = df['opening_efficiency_momentum'] * df['closing_efficiency_fractal']
    
    # Amount-Volume Efficiency
    df['amount_volume_efficiency'] = (df['amount'] / df['amount'].shift(1)) * df['efficiency']
    
    # Efficiency-Volume Scaling
    df['efficiency_volume_scaling'] = df['efficiency'] * df['volume'] / (df['efficiency'].shift(1) * df['volume'].shift(1))
    
    # Liquidity Efficiency Fractal
    df['liquidity_efficiency_fractal'] = ((df['close'] - (df['high'] * df['volume'] + df['low'] * df['volume']) / (df['volume'] * 2)) / df['range']) * df['volume']
    
    # Bullish Microstructure Divergence
    current_rejection_eff = (df['upper_rejection'] - df['lower_rejection']) / df['range']
    prev_rejection_eff = (df['upper_rejection'].shift(1) - df['lower_rejection'].shift(1)) / df['range'].shift(1)
    df['bullish_microstructure_divergence'] = (current_rejection_eff - prev_rejection_eff) * (df['efficiency'] - df['efficiency'].shift(1))
    
    # Bearish Microstructure Divergence
    df['bearish_microstructure_divergence'] = (prev_rejection_eff - current_rejection_eff) * (df['efficiency'].shift(1) - df['efficiency'])
    
    # Divergence Momentum
    df['divergence_momentum'] = df['bullish_microstructure_divergence'] - df['bearish_microstructure_divergence']
    
    # Short-Medium Term Rejection Divergence
    df['short_medium_term_rejection_divergence'] = df['short_term_rejection_momentum'] * df['medium_term_rejection_acceleration']
    
    # Volatility-Efficiency Divergence
    df['volatility_efficiency_divergence'] = (df['range'] / df['range'].shift(1)) * (df['efficiency'] - df['efficiency'].shift(1))
    
    # Volume-Regime Divergence
    df['volume_regime_divergence'] = (df['volume'] / df['volume_ma_4']) * (df['upper_rejection'] - df['lower_rejection'])
    
    # Rejection-Volume Coherence
    df['rejection_volume_coherence'] = (df['upper_rejection'] - df['lower_rejection']) * (df['volume'] / df['volume'].shift(1))
    
    # Efficiency-Momentum Alignment
    df['efficiency_momentum_alignment'] = np.sign(df['efficiency'] - df['efficiency'].shift(1)) * np.sign(df['close'] - df['close'].shift(1))
    
    # Regime-Transition Confirmation
    df['regime_transition_confirmation'] = df['volatility_rejection_persistence'] * df['volume_rejection_regime']
    
    # Core Microstructure Signals
    df['primary_rejection_momentum'] = df['directional_rejection_pressure'] * df['volume_confirmed_rejection']
    df['regime_transition_signal'] = df['high_volatility_rejection'] * df['volume_surge_rejection']
    df['efficiency_enhancement'] = df['gap_rejection_efficiency'] * df['amount_volume_efficiency']
    
    # Validation Layers
    df['microstructure_coherence'] = df['rejection_volume_coherence'] * df['efficiency_momentum_alignment']
    df['regime_persistence'] = df['volatility_rejection_persistence'] * df['volume_rejection_regime']
    df['pattern_confidence'] = df['regime_transition_confirmation'] * df['microstructure_coherence']
    
    # Final Alpha Synthesis
    df['weighted_core_signal'] = df['primary_rejection_momentum'] * df['microstructure_coherence']
    df['enhanced_regime_factor'] = df['regime_transition_signal'] * df['regime_persistence']
    df['validated_efficiency'] = df['efficiency_enhancement'] * df['pattern_confidence']
    
    # Composite Fractal Alpha
    df['composite_fractal_alpha'] = (df['weighted_core_signal'] * df['enhanced_regime_factor'] * 
                                    df['validated_efficiency'] * df['divergence_momentum'])
    
    # Dynamic Weighting
    df['dynamic_weighting'] = df['pattern_confidence'] * df['microstructure_coherence']
    
    # Regime Adaptation
    df['regime_adaptation'] = df['regime_persistence'] * df['volatility_rejection_persistence']
    
    # Final Adaptive Alpha
    df['final_adaptive_alpha'] = df['composite_fractal_alpha'] * df['dynamic_weighting'] * df['regime_adaptation']
    
    # Clean up intermediate columns
    result = df['final_adaptive_alpha'].copy()
    
    return result
