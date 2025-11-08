import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Momentum-Volume Divergence Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Components
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum Components
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_20d'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Combined Price Momentum (weighted average)
    data['price_momentum'] = (0.5 * data['price_momentum_5d'] + 
                             0.3 * data['price_momentum_10d'] + 
                             0.2 * data['price_momentum_20d'])
    
    # Combined Volume Momentum (weighted average)
    data['volume_momentum'] = (0.5 * data['volume_momentum_5d'] + 
                              0.3 * data['volume_momentum_10d'] + 
                              0.2 * data['volume_momentum_20d'])
    
    # Regime Detection
    data['amount_20d_avg'] = data['amount'].rolling(window=20).mean()
    data['amount_momentum'] = data['amount'] / data['amount'].shift(5) - 1
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['avg_price_range'] = data['price_range'].rolling(window=20).mean()
    
    # Regime Classification
    data['amount_regime'] = np.where(data['amount_momentum'] > 0.1, 'high',
                                   np.where(data['amount_momentum'] < -0.1, 'low', 'neutral'))
    data['volatility_regime'] = np.where(data['price_range'] > data['avg_price_range'] * 1.2, 'high', 'low')
    
    # Exponential Smoothing
    alpha = 0.3
    data['smoothed_price_momentum'] = data['price_momentum'].ewm(alpha=alpha).mean()
    data['smoothed_volume_momentum'] = data['volume_momentum'].ewm(alpha=alpha).mean()
    
    # Momentum Divergence
    data['momentum_divergence'] = data['smoothed_price_momentum'] - data['smoothed_volume_momentum']
    data['divergence_magnitude'] = abs(data['momentum_divergence'])
    
    # Cross-sectional ranking (within each date)
    data['divergence_rank'] = data.groupby(data.index)['momentum_divergence'].rank(pct=True)
    
    # Regime-based weighting
    def regime_weight(row):
        if row['amount_regime'] == 'high':
            # Emphasize recent divergence
            weight = 0.7 * row['momentum_divergence'] + 0.3 * row['smoothed_price_momentum']
        elif row['amount_regime'] == 'low':
            # Emphasize smoothed divergence
            weight = 0.3 * row['momentum_divergence'] + 0.7 * row['smoothed_price_momentum']
        else:  # neutral
            weight = 0.5 * row['momentum_divergence'] + 0.5 * row['smoothed_price_momentum']
        
        # Volatility adjustment
        if row['volatility_regime'] == 'high':
            weight *= (1 + row['price_range'])
        else:
            weight *= (1 + row['divergence_magnitude'])
        
        return weight
    
    data['regime_weighted_divergence'] = data.apply(regime_weight, axis=1)
    
    # Factor Construction
    def factor_construction(row):
        if row['amount_regime'] == 'high':
            if row['momentum_divergence'] > 0:
                # High amount + positive divergence: strong bullish
                factor = 1.5 * row['regime_weighted_divergence']
            else:
                # High amount + negative divergence: strong bearish
                factor = 1.5 * row['regime_weighted_divergence']
        elif row['amount_regime'] == 'low':
            if row['momentum_divergence'] > 0:
                # Low amount + positive divergence: weak bullish
                factor = 0.7 * row['regime_weighted_divergence']
            else:
                # Low amount + negative divergence: weak bearish
                factor = 0.7 * row['regime_weighted_divergence']
        else:  # neutral
            factor = row['regime_weighted_divergence']
        
        # Final scaling
        if row['volatility_regime'] == 'high':
            factor *= (1 + row['price_range'])
        else:
            # Scale by divergence persistence (rolling correlation)
            if len(data) > 10:
                recent_div = data['momentum_divergence'].tail(10)
                persistence = recent_div.autocorr() if not pd.isna(recent_div.autocorr()) else 0
                factor *= (1 + abs(persistence))
        
        return factor
    
    data['alpha_factor'] = data.apply(factor_construction, axis=1)
    
    # Stationary measure (z-score normalization)
    data['alpha_factor_stationary'] = (data['alpha_factor'] - data['alpha_factor'].rolling(window=20).mean()) / data['alpha_factor'].rolling(window=20).std()
    
    # Return the final alpha factor
    return data['alpha_factor_stationary']
