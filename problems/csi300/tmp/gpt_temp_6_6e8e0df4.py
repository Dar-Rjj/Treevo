import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate composite alpha factor using momentum acceleration, volume divergence, 
    and volatility regime analysis with non-linear interactions.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate multi-timeframe returns
    df['ultra_short'] = df['close'] / df['close'].shift(1) - 1
    df['short_term'] = df['close'] / df['close'].shift(3) - 1
    df['medium_term'] = df['close'] / df['close'].shift(8) - 1
    df['long_term'] = df['close'] / df['close'].shift(15) - 1
    
    # Acceleration structure
    df['primary_acceleration'] = df['short_term'] - df['medium_term']
    df['secondary_acceleration'] = df['medium_term'] - df['long_term']
    df['acceleration_curvature'] = df['primary_acceleration'] - df['secondary_acceleration']
    df['momentum_jerk'] = df['acceleration_curvature'] - df['acceleration_curvature'].shift(1)
    
    # Acceleration regimes
    df['strong_acceleration'] = ((df['primary_acceleration'] > 0.02) & 
                                (df['secondary_acceleration'] > 0.01)).astype(float)
    df['deceleration'] = ((df['primary_acceleration'] < -0.01) & 
                         (df['secondary_acceleration'] < 0)).astype(float)
    df['acceleration_reversal'] = (np.sign(df['primary_acceleration']) != 
                                  np.sign(df['secondary_acceleration'])).astype(float)
    
    # Calculate acceleration persistence (3-day consistency)
    df['acceleration_sign'] = np.sign(df['primary_acceleration'])
    df['acceleration_persistence'] = df['acceleration_sign'].rolling(window=3).apply(
        lambda x: len(set(x)) == 1 if not x.isnull().any() else np.nan, raw=False
    ).fillna(0)
    
    # Volume dynamics
    df['volume_change'] = df['volume'] / df['volume'].shift(1) - 1
    df['volume_trend'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_acceleration'] = df['volume_change'] - df['volume_change'].shift(1)
    
    # Volume persistence (positive volume change over last 3 days)
    df['volume_persistence'] = df['volume_change'].rolling(window=3).apply(
        lambda x: (x > 0).sum() if not x.isnull().any() else np.nan, raw=False
    ).fillna(0)
    
    # Price-volume divergence
    df['directional_divergence'] = (np.sign(df['short_term']) != 
                                   np.sign(df['volume_change'])).astype(float)
    df['magnitude_divergence'] = np.abs(df['short_term']) - np.abs(df['volume_change'])
    df['acceleration_divergence'] = df['primary_acceleration'] * df['volume_acceleration']
    df['divergence_strength'] = df['directional_divergence'] * df['magnitude_divergence']
    
    # Volume regime signals
    df['volume_confirmation'] = (df['volume_change'] * df['short_term'] > 0).astype(float)
    df['volume_exhaustion'] = ((df['volume_persistence'] > 2) & 
                              (df['volume_acceleration'] < 0)).astype(float)
    df['volume_breakout'] = (df['volume_acceleration'] > 0.5 * df['primary_acceleration']).astype(float)
    df['volume_divergence_alert'] = (df['divergence_strength'] > 0.01).astype(float)
    
    # Volatility measures
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['short_term_volatility'] = df['daily_range'].rolling(window=3).std()
    df['medium_term_volatility'] = df['daily_range'].rolling(window=8).std()
    df['volatility_ratio'] = df['short_term_volatility'] / df['medium_term_volatility']
    
    # Volatility regimes
    df['high_volatility'] = (df['volatility_ratio'] > 1.4).astype(float)
    df['low_volatility'] = (df['volatility_ratio'] < 0.6).astype(float)
    df['expanding_volatility'] = ((df['volatility_ratio'] > 1.2) & 
                                 (df['volatility_ratio'] > df['volatility_ratio'].shift(1))).astype(float)
    df['contracting_volatility'] = ((df['volatility_ratio'] < 0.8) & 
                                   (df['volatility_ratio'] < df['volatility_ratio'].shift(1))).astype(float)
    
    # Volatility regime persistence
    df['volatility_regime'] = np.select(
        [df['high_volatility'] == 1, df['low_volatility'] == 1],
        [2, 1],
        default=0
    )
    df['volatility_regime_persistence'] = df['volatility_regime'].rolling(window=3).apply(
        lambda x: len(set(x)) == 1 if not x.isnull().any() else np.nan, raw=False
    ).fillna(0)
    
    df['regime_strength'] = df['volatility_regime_persistence'] * df['volatility_ratio']
    df['regime_stability'] = 1 - np.abs(df['volatility_ratio'] - 1)
    
    # Non-linear interactions
    # Momentum-volume interactions
    df['acceleration_with_volume'] = df['primary_acceleration'] * df['volume_persistence']
    df['momentum_divergence_penalty'] = df['short_term'] * (1 - np.abs(df['volume_change']))
    df['volume_confirmed_acceleration'] = df['primary_acceleration'] * df['volume_confirmation']
    df['volume_exhaustion_signal'] = df['acceleration_curvature'] * df['volume_exhaustion']
    
    # Momentum-volatility interactions
    df['volatility_adjusted_acceleration'] = df['primary_acceleration'] / (1 + df['short_term_volatility'])
    df['regime_persistent_momentum'] = df['short_term'] * df['volatility_regime_persistence']
    df['high_vol_acceleration'] = df['primary_acceleration'] * df['high_volatility']
    df['low_vol_momentum_quality'] = df['short_term'] * df['low_volatility']
    
    # Three-way regime interactions
    df['strong_trend_confirmation'] = (df['strong_acceleration'] * 
                                      df['volume_confirmation'] * 
                                      df['low_volatility'])
    df['acceleration_breakout'] = (df['acceleration_reversal'] * 
                                  df['volume_breakout'] * 
                                  df['expanding_volatility'])
    df['divergence_warning'] = (df['deceleration'] * 
                               df['volume_divergence_alert'] * 
                               df['high_volatility'])
    df['regime_stability_premium'] = (df['acceleration_persistence'] * 
                                     df['regime_stability'] * 
                                     df['volume_persistence'])
    
    # Composite alpha construction
    # Base factor components
    df['core_acceleration'] = df['volatility_adjusted_acceleration']
    df['volume_aligned'] = df['core_acceleration'] * (1 + 0.3 * df['volume_confirmation'])
    df['divergence_adjusted'] = df['volume_aligned'] * (1 - 0.4 * df['divergence_warning'])
    df['regime_enhanced'] = df['divergence_adjusted'] * (1 + 0.2 * df['strong_trend_confirmation'])
    
    # Non-linear enhancement
    df['acceleration_weighted'] = df['regime_enhanced'] * (1 + 0.15 * df['momentum_jerk'])
    df['volume_breakout_boost'] = df['acceleration_weighted'] + 0.25 * df['acceleration_breakout']
    df['stability_premium'] = df['volume_breakout_boost'] * df['regime_stability_premium']
    df['persistence_filter'] = df['stability_premium'] * df['acceleration_persistence']
    
    # Final alpha output
    result = df['persistence_filter'].copy()
    
    # Clean up intermediate columns
    columns_to_drop = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'amount', 'volume']]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    return result
