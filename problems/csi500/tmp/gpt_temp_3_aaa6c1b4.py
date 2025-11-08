import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Momentum-Volume Divergence Alpha Factor
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Price Momentum Components
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum Components
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_20d'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Combined momentum scores (weighted average)
    data['price_momentum_combined'] = (
        0.5 * data['price_momentum_5d'] + 
        0.3 * data['price_momentum_10d'] + 
        0.2 * data['price_momentum_20d']
    )
    data['volume_momentum_combined'] = (
        0.5 * data['volume_momentum_5d'] + 
        0.3 * data['volume_momentum_10d'] + 
        0.2 * data['volume_momentum_20d']
    )
    
    # Exponential Smoothing
    alpha_smooth = 0.3
    data['price_momentum_ema'] = data['price_momentum_combined'].ewm(alpha=alpha_smooth).mean()
    data['volume_momentum_ema'] = data['volume_momentum_combined'].ewm(alpha=alpha_smooth).mean()
    
    # Momentum Acceleration
    data['price_momentum_accel'] = data['price_momentum_ema'] - data['price_momentum_ema'].shift(1)
    data['volume_momentum_accel'] = data['volume_momentum_ema'] - data['volume_momentum_ema'].shift(1)
    
    # Regime Detection using Amount Data
    data['amount_momentum_10d'] = data['amount'] / data['amount'].shift(10) - 1
    data['amount_acceleration'] = data['amount_momentum_10d'] - data['amount_momentum_10d'].shift(1)
    
    # Regime Classification
    def classify_regime(row):
        if pd.isna(row['amount_momentum_10d']) or pd.isna(row['amount_acceleration']):
            return np.nan
        elif row['amount_momentum_10d'] > 0.15 and row['amount_acceleration'] > 0.05:
            return 2  # High Activity Regime
        elif row['amount_momentum_10d'] < -0.05 and row['amount_acceleration'] < -0.02:
            return 0  # Low Activity Regime
        else:
            return 1  # Transition Regime
    
    data['regime'] = data.apply(classify_regime, axis=1)
    
    # Divergence Pattern Analysis
    def calculate_divergence(row):
        if pd.isna(row['price_momentum_ema']) or pd.isna(row['volume_momentum_ema']):
            return np.nan
        
        price_dir = 1 if row['price_momentum_ema'] > 0 else -1
        volume_dir = 1 if row['volume_momentum_ema'] > 0 else -1
        
        # Directional Divergence
        if price_dir == 1 and volume_dir == -1:
            directional_score = -1.0  # Bearish divergence
        elif price_dir == -1 and volume_dir == 1:
            directional_score = 1.0   # Bullish divergence
        else:
            directional_score = 0.5 * price_dir  # Confirmation
            
        # Magnitude Divergence
        price_mag = abs(row['price_momentum_ema'])
        volume_mag = abs(row['volume_momentum_ema'])
        
        if price_mag > 0.05 and volume_mag < 0.02:
            magnitude_score = -0.5  # Weak move
        elif price_mag < 0.02 and volume_mag > 0.05:
            magnitude_score = 0.5   # Strong move pending
        else:
            magnitude_score = 0.0   # Valid trend
            
        return directional_score + magnitude_score
    
    data['divergence_raw'] = data.apply(calculate_divergence, axis=1)
    
    # Regime-Adaptive Weighting
    def regime_weighted_score(row):
        if pd.isna(row['divergence_raw']) or pd.isna(row['regime']):
            return np.nan
            
        if row['regime'] == 2:  # High Activity Regime
            # Higher weight to volume confirmation
            volume_weight = 0.7
            price_weight = 0.3
        elif row['regime'] == 0:  # Low Activity Regime
            # Higher weight to price momentum
            volume_weight = 0.3
            price_weight = 0.7
        else:  # Transition Regime
            volume_weight = 0.5
            price_weight = 0.5
            
        return (price_weight * row['price_momentum_ema'] + 
                volume_weight * row['volume_momentum_ema'] + 
                row['divergence_raw'])
    
    data['regime_weighted_score'] = data.apply(regime_weighted_score, axis=1)
    
    # Cross-Sectional Ranking
    # For demonstration, we'll use rolling percentile ranking
    data['alpha_factor'] = data['regime_weighted_score'].rolling(
        window=20, min_periods=10
    ).apply(lambda x: (x.rank(pct=True).iloc[-1] - 0.5) * 2 if len(x.dropna()) >= 10 else np.nan)
    
    return data['alpha_factor']
