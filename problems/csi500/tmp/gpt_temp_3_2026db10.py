import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Price-Volume Momentum Divergence factor
    """
    df = data.copy()
    
    # Price Momentum Calculation
    df['price_mom_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_mom_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_mom_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Momentum Calculation
    df['volume_mom_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_mom_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_mom_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Volatility Regime Detection
    df['daily_range_ratio'] = (df['high'] - df['low']) / df['close']
    df['volatility_5d'] = df['daily_range_ratio'].rolling(window=5).mean()
    df['volatility_20d_avg'] = df['daily_range_ratio'].rolling(window=20).mean()
    
    # Regime Classification
    conditions = [
        df['volatility_5d'] > (1.5 * df['volatility_20d_avg']),
        (df['volatility_5d'] >= (0.8 * df['volatility_20d_avg'])) & 
        (df['volatility_5d'] <= (1.5 * df['volatility_20d_avg'])),
        df['volatility_5d'] < (0.8 * df['volatility_20d_avg'])
    ]
    choices = ['high', 'normal', 'low']
    df['volatility_regime'] = np.select(conditions, choices, default='normal')
    
    # Raw Divergence Calculation
    df['divergence_5d'] = df['price_mom_5d'] - df['volume_mom_5d']
    df['divergence_10d'] = df['price_mom_10d'] - df['volume_mom_10d']
    df['divergence_20d'] = df['price_mom_20d'] - df['volume_mom_20d']
    
    # Signal Quality Assessment
    divergence_cols = ['divergence_5d', 'divergence_10d', 'divergence_20d']
    df['direction_consistency'] = df[divergence_cols].apply(
        lambda x: sum(x > 0) if all(x > 0) else sum(x < 0) if all(x < 0) else 0, axis=1
    )
    
    df['strength_consistency'] = df[divergence_cols].var(axis=1)
    df['quality_score'] = df['direction_consistency'] * (1 - df['strength_consistency'])
    
    # Regime-Specific Weighting Scheme
    def get_weights(regime):
        if regime == 'high':
            price_weights = [0.2, 0.3, 0.5]  # short, medium, long
            volume_weights = [0.4, 0.4, 0.2]
        elif regime == 'normal':
            price_weights = [0.3, 0.4, 0.3]
            volume_weights = [0.3, 0.4, 0.3]
        else:  # low volatility
            price_weights = [0.4, 0.4, 0.2]
            volume_weights = [0.2, 0.3, 0.5]
        return price_weights, volume_weights
    
    # Weighted Momentum Calculation
    price_moms = ['price_mom_5d', 'price_mom_10d', 'price_mom_20d']
    volume_moms = ['volume_mom_5d', 'volume_mom_10d', 'volume_mom_20d']
    
    df['weighted_price_mom'] = 0.0
    df['weighted_volume_mom'] = 0.0
    
    for idx, row in df.iterrows():
        price_weights, volume_weights = get_weights(row['volatility_regime'])
        
        weighted_price = sum(row[price_moms].fillna(0) * price_weights)
        weighted_volume = sum(row[volume_moms].fillna(0) * volume_weights)
        
        df.loc[idx, 'weighted_price_mom'] = weighted_price
        df.loc[idx, 'weighted_volume_mom'] = weighted_volume
    
    df['final_divergence'] = df['weighted_price_mom'] - df['weighted_volume_mom']
    
    # Quality Enhancement
    df['enhanced_divergence'] = df['final_divergence'] * df['quality_score']
    
    # Regime intensity adjustment
    df['regime_intensity'] = abs(df['volatility_5d'] / df['volatility_20d_avg'] - 1)
    
    # Final alpha factor
    df['alpha_factor'] = df['enhanced_divergence'] * (1 + df['regime_intensity'])
    
    return df['alpha_factor']
