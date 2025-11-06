import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-scale acceleration framework with cross-dimensional interactions
    """
    # Price Acceleration Structure
    df['ultra_short_acceleration'] = (df['close']/df['close'].shift(1)-1) - (df['close'].shift(1)/df['close'].shift(2)-1)
    df['short_term_acceleration'] = (df['close']/df['close'].shift(3)-1) - (df['close'].shift(1)/df['close'].shift(4)-1)
    df['medium_term_acceleration'] = (df['close']/df['close'].shift(8)-1) - (df['close'].shift(2)/df['close'].shift(10)-1)
    df['acceleration_curvature'] = df['ultra_short_acceleration'] - 2*df['short_term_acceleration'] + df['medium_term_acceleration']
    
    # Volume Acceleration Dynamics
    df['volume_momentum'] = df['volume']/df['volume'].shift(1)-1
    df['volume_acceleration'] = df['volume_momentum'] - df['volume_momentum'].shift(1)
    df['volume_trend_acceleration'] = (df['volume']/df['volume'].shift(3)-1) - (df['volume'].shift(1)/df['volume'].shift(4)-1)
    df['volume_curvature'] = df['volume_momentum'] - 2*df['volume_acceleration'] + df['volume_trend_acceleration']
    
    # Volatility Acceleration Metrics
    df['range_acceleration'] = ((df['high']-df['low'])/df['close']) - ((df['high'].shift(1)-df['low'].shift(1))/df['close'].shift(1))
    
    short_term_vol = df['close'].pct_change().rolling(window=5).std()
    medium_term_vol = df['close'].pct_change().rolling(window=20).std()
    df['volatility_regime_acceleration'] = (short_term_vol/medium_term_vol) - (short_term_vol.shift(1)/medium_term_vol.shift(1))
    
    df['intraday_efficiency'] = (df['close']-df['low'])/(df['high']-df['low'])
    df['intraday_efficiency_acceleration'] = df['intraday_efficiency'] - df['intraday_efficiency'].shift(1)
    df['volatility_curvature'] = df['range_acceleration'] - 2*df['volatility_regime_acceleration'] + df['intraday_efficiency_acceleration']
    
    # Multi-Dimensional Regime Persistence
    def count_consecutive_same_sign(series):
        signs = np.sign(series)
        persistence = []
        current_count = 0
        for i in range(len(signs)):
            if i == 0 or signs[i] == signs[i-1]:
                current_count += 1
            else:
                current_count = 1
            persistence.append(current_count)
        return pd.Series(persistence, index=series.index)
    
    df['price_acceleration_persistence'] = count_consecutive_same_sign(df['ultra_short_acceleration'])
    df['volume_acceleration_persistence'] = count_consecutive_same_sign(df['volume_acceleration'])
    df['volatility_acceleration_persistence'] = count_consecutive_same_sign(df['volatility_regime_acceleration'])
    
    # Cross-acceleration persistence
    all_same_sign = ((np.sign(df['ultra_short_acceleration']) == np.sign(df['volume_acceleration'])) & 
                    (np.sign(df['ultra_short_acceleration']) == np.sign(df['volatility_regime_acceleration'])))
    df['cross_acceleration_persistence'] = count_consecutive_same_sign(all_same_sign.astype(int))
    
    # Price-Volume Alignment Persistence
    price_volume_same = (np.sign(df['ultra_short_acceleration']) == np.sign(df['volume_acceleration']))
    df['confirmation_persistence'] = count_consecutive_same_sign(price_volume_same.astype(int))
    
    price_volume_opposite = (np.sign(df['ultra_short_acceleration']) != np.sign(df['volume_acceleration']))
    df['divergence_persistence'] = count_consecutive_same_sign(price_volume_opposite.astype(int))
    
    df['strong_alignment_persistence'] = count_consecutive_same_sign(all_same_sign.astype(int))
    
    # Regime Quality Metrics
    df['acceleration_stability'] = pd.Series([np.sign(df['ultra_short_acceleration'].iloc[max(0,i-4):i+1]).var() 
                                            for i in range(len(df))], index=df.index)
    
    df['alignment_consistency'] = df['price_acceleration_persistence'].rolling(window=10).corr(df['volume_acceleration_persistence'])
    
    # Regime transition probability (simplified)
    regime_changes = (np.sign(df['ultra_short_acceleration']) != np.sign(df['ultra_short_acceleration'].shift(1))).astype(int)
    df['regime_transition_probability'] = regime_changes.rolling(window=10).mean()
    
    # Multi-scale persistence score
    persistence_weights = [0.4, 0.3, 0.2, 0.1]  # weights for different persistence types
    df['multi_scale_persistence_score'] = (
        df['price_acceleration_persistence'] * persistence_weights[0] +
        df['volume_acceleration_persistence'] * persistence_weights[1] +
        df['volatility_acceleration_persistence'] * persistence_weights[2] +
        df['cross_acceleration_persistence'] * persistence_weights[3]
    )
    
    # Cross-Dimensional Interaction Engine
    df['price_volume_acceleration_coupling'] = df['ultra_short_acceleration'] * df['volume_acceleration']
    df['price_volatility_acceleration_coupling'] = df['ultra_short_acceleration'] * df['volatility_regime_acceleration']
    df['volume_volatility_acceleration_coupling'] = df['volume_acceleration'] * df['volatility_regime_acceleration']
    df['triple_acceleration_product'] = (df['ultra_short_acceleration'] * df['volume_acceleration'] * 
                                       df['volatility_regime_acceleration'])
    
    # Persistence-Enhanced Interactions
    df['persistent_acceleration_momentum'] = df['ultra_short_acceleration'] * df['price_acceleration_persistence']
    df['volume_confirmed_acceleration'] = df['ultra_short_acceleration'] * df['confirmation_persistence']
    df['volatility_stabilized_acceleration'] = df['ultra_short_acceleration'] * df['volatility_acceleration_persistence']
    df['multi_persistence_weighted_acceleration'] = df['triple_acceleration_product'] * df['multi_scale_persistence_score']
    
    # Regime Transition Interactions
    df['acceleration_regime_change'] = df['acceleration_curvature'] * df['regime_transition_probability']
    df['persistence_breakdown'] = df['multi_persistence_weighted_acceleration'] * (1 - df['acceleration_stability'])
    df['alignment_shift'] = df['strong_alignment_persistence'] * df['acceleration_regime_change']
    df['cross_dimensional_momentum'] = df['persistence_breakdown'] * df['alignment_shift']
    
    # Adaptive Composite Construction
    # Base Acceleration Components
    df['core_price_acceleration'] = df['ultra_short_acceleration'] * df['price_acceleration_persistence']
    df['volume_enhanced_acceleration'] = df['core_price_acceleration'] * (1 + df['volume_acceleration_persistence'])
    df['volatility_adjusted_acceleration'] = df['volume_enhanced_acceleration'] * (2 - abs(df['volatility_regime_acceleration']))
    df['multi_scale_acceleration_composite'] = df['volatility_adjusted_acceleration'] * df['acceleration_curvature']
    
    # Persistence-Weighted Enhancement
    df['cross_dimensional_persistence'] = df['multi_scale_acceleration_composite'] * df['multi_scale_persistence_score']
    df['alignment_persistence_boost'] = df['cross_dimensional_persistence'] * df['strong_alignment_persistence']
    df['regime_stability_overlay'] = df['alignment_persistence_boost'] * df['acceleration_stability']
    df['transition_aware_adjustment'] = df['regime_stability_overlay'] * (1 - df['regime_transition_probability'])
    
    # Interaction Finalization
    df['acceleration_interaction_multiplier'] = df['transition_aware_adjustment'] * df['triple_acceleration_product']
    df['persistence_interaction_finalizer'] = df['acceleration_interaction_multiplier'] * df['cross_acceleration_persistence']
    df['regime_quality_overlay'] = df['persistence_interaction_finalizer'] * df['alignment_consistency']
    df['final_alpha'] = df['regime_quality_overlay'] * df['cross_dimensional_momentum']
    
    # Handle NaN values
    df['final_alpha'] = df['final_alpha'].fillna(0)
    
    return df['final_alpha']
