import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor using multi-timeframe momentum, volatility regimes, 
    volume confirmation, and cross-regime interactions.
    """
    # Multi-Timeframe Momentum Structure
    df['ultra_short_ret'] = df['close'] / df['close'].shift(1) - 1
    df['short_term_ret'] = df['close'] / df['close'].shift(3) - 1
    df['medium_term_ret'] = df['close'] / df['close'].shift(8) - 1
    df['long_term_ret'] = df['close'] / df['close'].shift(15) - 1
    
    # Momentum Acceleration Patterns
    df['primary_acceleration'] = df['short_term_ret'] - df['medium_term_ret']
    df['secondary_acceleration'] = df['medium_term_ret'] - df['long_term_ret']
    df['curvature'] = df['short_term_ret'] - 2 * df['medium_term_ret'] + df['long_term_ret']
    df['momentum_regime'] = np.sign(df['short_term_ret']) * np.sign(df['medium_term_ret']) * np.sign(df['long_term_ret'])
    
    # Momentum Quality Metrics
    df['momentum_consistency'] = ((df['ultra_short_ret'] > 0) & (df['short_term_ret'] > 0) & (df['medium_term_ret'] > 0)).astype(int)
    df['strength_ratio'] = np.abs(df['short_term_ret']) / (np.abs(df['medium_term_ret']) + 1e-8)
    df['decay_profile'] = df['short_term_ret'] / (df['long_term_ret'] + 1e-8)
    df['momentum_persistence'] = df['close'].rolling(5).apply(lambda x: (x.pct_change().dropna() > 0).sum(), raw=False)
    
    # Volatility Regime Framework
    df['daily_range_vol'] = (df['high'] - df['low']) / df['close']
    df['short_term_vol'] = df['daily_range_vol'].rolling(3).std()
    df['medium_term_vol'] = df['daily_range_vol'].rolling(8).std()
    df['volatility_ratio'] = df['short_term_vol'] / (df['medium_term_vol'] + 1e-8)
    
    # Volatility State Classification
    df['high_volatility'] = (df['volatility_ratio'] > 1.3).astype(int)
    df['low_volatility'] = (df['volatility_ratio'] < 0.7).astype(int)
    df['expanding_volatility'] = (df['volatility_ratio'] > 1.1).astype(int)
    df['contracting_volatility'] = (df['volatility_ratio'] < 0.9).astype(int)
    
    # Volatility-Adjusted Signals
    df['risk_adjusted_momentum'] = df['short_term_ret'] / (df['daily_range_vol'] + 1e-8)
    df['volatility_persistence'] = df['high_volatility'].rolling(3).sum()
    df['volatility_breakout'] = (df['volatility_ratio'] > 1.5).astype(int) * df['primary_acceleration']
    df['volatility_stability'] = df['volatility_persistence'] * (1 / (df['volatility_ratio'] + 1e-8))
    
    # Volume Confirmation System
    df['volume_momentum'] = df['volume'] / df['volume'].shift(1) - 1
    df['volume_trend'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_acceleration'] = df['volume_momentum'] - df['volume_momentum'].shift(1)
    df['volume_persistence'] = (df['volume_momentum'] > 0).rolling(3).sum()
    
    # Price-Volume Interaction
    df['confirmation_strength'] = np.sign(df['short_term_ret']) * np.sign(df['volume_momentum'])
    df['divergence_detection'] = (np.sign(df['short_term_ret']) != np.sign(df['volume_momentum'])).astype(int)
    df['divergence_magnitude'] = np.abs(df['short_term_ret']) * np.abs(df['volume_momentum'])
    df['volume_regime'] = (df['volume'] > 1.5 * df['volume'].shift(1)).astype(int)
    
    # Volume Quality Assessment
    df['volume_consistency'] = (np.sign(df['volume_momentum']) == np.sign(df['volume_momentum'].shift(1))).rolling(3).sum()
    df['volume_to_price_ratio'] = df['volume_momentum'] / (np.abs(df['short_term_ret']) + 1e-8)
    df['high_volume_confirmation'] = df['volume_regime'] * df['confirmation_strength']
    df['low_volume_divergence'] = (df['volume_regime'] == 0).astype(int) * df['divergence_magnitude']
    
    # Cross-Regime Interaction Engine
    # Momentum-Volatility Interactions
    df['high_momentum_low_vol'] = df['primary_acceleration'] * (1 / (df['volatility_ratio'] + 1e-8))
    df['low_momentum_high_vol'] = df['curvature'] * df['volatility_ratio']
    df['regime_aligned_momentum'] = df['risk_adjusted_momentum'] * df['volatility_persistence']
    df['volatility_breakout_momentum'] = df['volatility_breakout'] * df['momentum_consistency']
    
    # Volume-Momentum Interactions
    df['confirmed_acceleration'] = df['primary_acceleration'] * df['confirmation_strength']
    df['divergence_acceleration'] = df['curvature'] * df['divergence_magnitude']
    df['volume_regime_momentum'] = df['momentum_persistence'] * df['volume_persistence']
    df['high_volume_breakout'] = df['volume_regime'] * df['risk_adjusted_momentum']
    
    # Three-Way Regime Alignment
    df['full_alignment'] = df['momentum_regime'] * df['volatility_stability'] * df['high_volume_confirmation']
    df['momentum_volume_alignment'] = df['momentum_consistency'] * df['volume_consistency']
    df['volatility_volume_alignment'] = df['volatility_persistence'] * df['volume_persistence']
    df['regime_conflict'] = (df['momentum_regime'] == -1).astype(int) * df['volatility_breakout'] * df['low_volume_divergence']
    
    # Composite Alpha Construction
    # Base Factor Components
    df['core_momentum'] = df['risk_adjusted_momentum']
    df['volume_enhanced'] = df['core_momentum'] * (1 + df['confirmation_strength'])
    df['volatility_scaled'] = df['volume_enhanced'] * (2 - df['volatility_ratio'])
    df['acceleration_overlay'] = df['volatility_scaled'] + df['primary_acceleration']
    
    # Regime-Adaptive Weighting
    df['alignment_boost'] = df['acceleration_overlay'] * df['full_alignment']
    df['divergence_adjustment'] = df['acceleration_overlay'] * (1 - np.abs(df['divergence_detection']))
    df['persistence_multiplier'] = df['acceleration_overlay'] * df['momentum_persistence']
    df['regime_stability_factor'] = df['acceleration_overlay'] * df['volatility_stability']
    
    # Final Alpha Output
    alpha_factor = (
        df['alignment_boost'] + 
        df['divergence_adjustment'] + 
        df['persistence_multiplier'] + 
        df['regime_stability_factor']
    ) / 4
    
    return alpha_factor
