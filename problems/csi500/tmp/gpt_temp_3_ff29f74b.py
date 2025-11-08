import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Regime-Adaptive Momentum Divergence Factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Vector Analysis
    # Price Momentum Vector Components
    data['mom_short'] = data['close'] / data['close'].shift(3) - 1
    data['mom_medium'] = data['close'] / data['close'].shift(5) - 1
    data['mom_long'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum Vector Geometry
    data['primary_angle'] = np.arctan(data['mom_medium'] / (data['mom_short'] + 1e-8))
    data['secondary_angle'] = np.arctan(data['mom_long'] / (data['mom_medium'] + 1e-8))
    data['momentum_divergence'] = np.abs(data['primary_angle'] - data['secondary_angle'])
    
    # Volume Momentum Analysis
    data['vol_mom_short'] = data['volume'] / data['volume'].shift(5) - 1
    data['vol_mom_long'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Price-Volume Divergence (5-day rolling correlation)
    data['price_vol_corr'] = data['mom_short'].rolling(window=5).corr(data['vol_mom_short'])
    data['price_volume_divergence'] = np.abs(data['price_vol_corr'])
    
    # Volatility Regime Classification
    # Intraday Volatility Structure (approximated using daily range)
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # Multi-Timeframe Volatility Assessment
    data['true_range'] = np.maximum(
        np.maximum(data['high'] - data['low'], 
                  np.abs(data['high'] - data['close'].shift(1))),
        np.abs(data['low'] - data['close'].shift(1))
    )
    data['vol_short'] = data['true_range'].rolling(window=5).mean()
    data['vol_medium'] = data['true_range'].rolling(window=10).mean()
    
    # Volatility Regime Classification
    def classify_volatility_regime(row):
        if (row['vol_short'] > row['vol_medium']) and (row['vol_short'] > 0.02):
            return 'high'
        elif (row['vol_short'] < row['vol_medium']) and (row['vol_short'] < 0.01):
            return 'low'
        else:
            return 'transitional'
    
    data['vol_regime'] = data.apply(classify_volatility_regime, axis=1)
    
    # Volume-Volatility Efficiency
    data['volume_efficiency'] = (data['high'] - data['low']) / (data['volume'] + 1e-8)
    
    # Volatility Clustering Detection
    data['high_vol_day'] = data['daily_range'] > data['daily_range'].rolling(window=20).mean()
    data['vol_cluster_count'] = data['high_vol_day'].rolling(window=5, min_periods=1).sum()
    
    # Regime-Adaptive Signal Generation
    # Momentum Regime Classification
    def classify_momentum_regime(row):
        if (row['mom_short'] > 0 and row['mom_medium'] > 0 and row['mom_long'] > 0 and 
            np.abs(row['primary_angle']) > np.radians(30) and np.abs(row['secondary_angle']) > np.radians(30)):
            return 'strong_trend'
        elif (row['mom_short'] > 0 and row['mom_medium'] > 0 and row['mom_long'] > 0 and 
              np.abs(row['primary_angle']) < np.radians(30)):
            return 'weak_trend'
        elif np.abs(row['primary_angle'] - row['secondary_angle']) > np.radians(45):
            return 'reversal'
        elif ((row['mom_short'] > 0 and row['mom_medium'] < 0) or 
              (row['mom_short'] < 0 and row['mom_medium'] > 0)):
            return 'mean_reversion'
        elif (np.abs(row['mom_short']) < 0.005 and np.abs(row['mom_medium']) < 0.005 and 
              np.abs(row['mom_long']) < 0.005):
            return 'consolidation'
        else:
            return 'mixed'
    
    data['momentum_regime'] = data.apply(classify_momentum_regime, axis=1)
    
    # Volume Confirmation
    def get_volume_confirmation(row):
        if ((row['mom_short'] > 0 and row['vol_mom_short'] > 0) or 
            (row['mom_short'] < 0 and row['vol_mom_short'] < 0)):
            return 'positive'
        else:
            return 'negative'
    
    data['volume_confirmation'] = data.apply(get_volume_confirmation, axis=1)
    
    # Hierarchical Factor Construction
    # Base Momentum Divergence Component
    data['momentum_vector_strength'] = np.sqrt(
        data['mom_short']**2 + data['mom_medium']**2 + data['mom_long']**2
    )
    
    # Weighted momentum divergence
    data['weighted_divergence'] = (
        data['momentum_divergence'] * 0.6 + 
        data['price_volume_divergence'] * 0.4
    )
    
    # Volatility Structure Component
    # Intraday Pattern Score (approximated)
    data['volatility_slope'] = data['daily_range'] / (data['daily_range'].shift(1) + 1e-8)
    data['intraday_pattern_score'] = np.where(
        data['volatility_slope'] > 1.2, 1.5,
        np.where(data['volatility_slope'] < 0.8, 0.7, 1.0)
    )
    
    # Volume Concentration Impact (approximated using daily volume pattern)
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['concentration_score'] = np.where(
        data['volume_concentration'] > 1.3, 1.4,
        np.where(data['volume_concentration'] < 0.7, 0.8, 1.0)
    )
    
    # Volatility Regime Base Score
    data['vol_regime_score'] = np.where(
        data['vol_regime'] == 'high', 1.2,
        np.where(data['vol_regime'] == 'low', 0.9, 1.0)
    )
    
    # Final Factor Synthesis
    # Base components
    base_momentum = data['momentum_vector_strength'] * data['weighted_divergence']
    volatility_structure = (data['intraday_pattern_score'] * 
                          data['concentration_score'] * 
                          data['vol_regime_score'])
    
    primary_factor = base_momentum * volatility_structure
    
    # Apply Regime-Adaptive Multipliers
    # Momentum regime multipliers
    momentum_multiplier = np.where(
        data['momentum_regime'] == 'strong_trend', 1.5,
        np.where(data['momentum_regime'] == 'weak_trend', 1.2,
        np.where(data['momentum_regime'] == 'reversal', 1.3,
        np.where(data['momentum_regime'] == 'mean_reversion', 1.4, 1.0)))
    )
    
    # Volatility regime multipliers
    volatility_multiplier = np.where(
        data['vol_regime'] == 'high', 1.2,
        np.where(data['vol_regime'] == 'low', 1.3, 1.0)
    )
    
    # Volume efficiency adjustments
    volume_efficiency_quantile = data['volume_efficiency'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    volume_efficiency_multiplier = np.where(
        volume_efficiency_quantile > 0.7, 1.3,
        np.where(volume_efficiency_quantile < 0.3, 0.7, 1.0)
    )
    
    # Volume confirmation multiplier
    volume_confirmation_multiplier = np.where(data['volume_confirmation'] == 'positive', 1.2, 0.8)
    
    # Volatility cluster adjustment
    cluster_multiplier = np.where(data['vol_cluster_count'] > 3, 1.4, 1.0)
    
    # Final factor calculation
    final_factor = (primary_factor * momentum_multiplier * volatility_multiplier * 
                   volume_efficiency_multiplier * volume_confirmation_multiplier * 
                   cluster_multiplier)
    
    return final_factor
