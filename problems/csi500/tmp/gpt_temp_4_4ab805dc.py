import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Momentum-Volume Divergence factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Calculation
    # Price Momentum
    data['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum
    data['volume_momentum_5'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_20'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Exponential Smoothing with alpha=0.3
    alpha = 0.3
    
    # Price Momentum EMA and Acceleration
    for period in [5, 10, 20]:
        price_col = f'price_momentum_{period}'
        ema_col = f'price_ema_{period}'
        acc_col = f'price_acc_{period}'
        
        data[ema_col] = data[price_col].ewm(alpha=alpha, adjust=False).mean()
        data[acc_col] = data[ema_col] - data[ema_col].shift(1)
    
    # Volume Momentum EMA and Acceleration
    for period in [5, 10, 20]:
        volume_col = f'volume_momentum_{period}'
        ema_col = f'volume_ema_{period}'
        acc_col = f'volume_acc_{period}'
        
        data[ema_col] = data[volume_col].ewm(alpha=alpha, adjust=False).mean()
        data[acc_col] = data[ema_col] - data[ema_col].shift(1)
    
    # Regime Detection using Amount Data
    data['amount_ma_10'] = data['amount'].rolling(window=10).mean()
    data['amount_volatility_20'] = data['amount'].rolling(window=20).std()
    
    # Regime shift detection using amount breakouts
    data['amount_zscore'] = (data['amount'] - data['amount_ma_10']) / data['amount_volatility_20'].replace(0, 1)
    
    # Regime classification
    def classify_regime(row):
        if pd.isna(row['amount_zscore']):
            return 'transition'
        if abs(row['amount_zscore']) > 2:
            return 'high_activity'
        elif abs(row['amount_zscore']) < 0.5:
            return 'low_activity'
        else:
            return 'transition'
    
    data['regime'] = data.apply(classify_regime, axis=1)
    
    # Divergence Analysis
    # Price-Volume Divergence
    divergence_scores = []
    
    for period in [5, 10, 20]:
        price_momentum = data[f'price_momentum_{period}']
        volume_momentum = data[f'volume_momentum_{period}']
        price_acc = data[f'price_acc_{period}']
        volume_acc = data[f'volume_acc_{period}']
        
        # Direction mismatch detection
        direction_divergence = np.sign(price_momentum) != np.sign(volume_momentum)
        
        # Magnitude discrepancy
        magnitude_divergence = abs(price_momentum - volume_momentum)
        
        # Acceleration divergence
        acc_divergence = abs(price_acc - volume_acc)
        
        # Combined divergence score for this timeframe
        timeframe_divergence = (direction_divergence.astype(float) * 0.4 + 
                               magnitude_divergence * 0.3 + 
                               acc_divergence * 0.3)
        
        divergence_scores.append(timeframe_divergence)
    
    # Multi-timeframe consistency scoring
    data['divergence_strength'] = pd.concat(divergence_scores, axis=1).mean(axis=1)
    
    # Factor Construction
    # Regime-Adaptive Weighting
    def calculate_regime_weighted_score(row):
        if pd.isna(row['divergence_strength']):
            return np.nan
            
        if row['regime'] == 'high_activity':
            # Higher volume weight (0.7)
            volume_weight = 0.7
            price_weight = 0.3
        elif row['regime'] == 'low_activity':
            # Higher price weight (0.7)
            price_weight = 0.7
            volume_weight = 0.3
        else:  # transition
            # Balanced weights (0.5 each)
            price_weight = 0.5
            volume_weight = 0.5
        
        # Calculate weighted divergence score
        short_term_div = (price_weight * row['price_momentum_5'] + 
                         volume_weight * row['volume_momentum_5'])
        medium_term_div = (price_weight * row['price_momentum_10'] + 
                          volume_weight * row['volume_momentum_10'])
        long_term_div = (price_weight * row['price_momentum_20'] + 
                        volume_weight * row['volume_momentum_20'])
        
        # Combine with divergence strength
        regime_score = (short_term_div * 0.4 + 
                       medium_term_div * 0.35 + 
                       long_term_div * 0.25) * row['divergence_strength']
        
        return regime_score
    
    data['regime_weighted_score'] = data.apply(calculate_regime_weighted_score, axis=1)
    
    # Cross-sectional ranking (within each date)
    def cross_sectional_rank(group):
        return group.rank(pct=True) - 0.5
    
    # Final factor scores with regime-specific adjustments
    data['final_factor'] = data.groupby(data.index)['regime_weighted_score'].transform(cross_sectional_rank)
    
    # Apply regime-specific final adjustments
    regime_multipliers = {
        'high_activity': 1.2,    # Amplify signals in high activity
        'low_activity': 0.8,     # Dampen signals in low activity
        'transition': 1.0        # No adjustment in transition
    }
    
    data['final_factor'] = data.apply(
        lambda row: row['final_factor'] * regime_multipliers.get(row['regime'], 1.0), 
        axis=1
    )
    
    return data['final_factor']
