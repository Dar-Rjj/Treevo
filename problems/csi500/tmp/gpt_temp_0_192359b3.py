import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Price Momentum Components
    data['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum Components
    data['volume_momentum_5'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_20'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Amount-Based Regime Detection
    data['amount_momentum_10'] = data['amount'] / data['amount'].shift(10) - 1
    data['amount_acceleration'] = (data['amount'] / data['amount'].shift(1)) - (data['amount'].shift(1) / data['amount'].shift(2))
    
    # Regime Classification
    data['regime'] = 'weak'
    data.loc[data['amount_momentum_10'].abs() > 0.2, 'regime'] = 'strong'
    data.loc[(data['amount_momentum_10'].abs() > 0.1) & (data['amount_momentum_10'].abs() <= 0.2), 'regime'] = 'moderate'
    
    # Momentum-Volume Divergence Analysis
    # Directional Divergence
    data['divergence_5'] = np.sign(data['price_momentum_5']) != np.sign(data['volume_momentum_5'])
    data['divergence_10'] = np.sign(data['price_momentum_10']) != np.sign(data['volume_momentum_10'])
    data['divergence_20'] = np.sign(data['price_momentum_20']) != np.sign(data['volume_momentum_20'])
    
    # Multi-timeframe consistency
    data['divergence_consistency'] = data[['divergence_5', 'divergence_10', 'divergence_20']].sum(axis=1)
    
    # Magnitude Divergence
    data['momentum_ratio_5'] = data['price_momentum_5'] / (data['volume_momentum_5'] + 1e-8)
    data['momentum_ratio_10'] = data['price_momentum_10'] / (data['volume_momentum_10'] + 1e-8)
    data['momentum_ratio_20'] = data['price_momentum_20'] / (data['volume_momentum_20'] + 1e-8)
    
    # Price and Volume Acceleration
    data['price_acceleration_5'] = data['price_momentum_5'] - data['price_momentum_5'].shift(1)
    data['volume_acceleration_5'] = data['volume_momentum_5'] - data['volume_momentum_5'].shift(1)
    data['acceleration_gap'] = data['price_acceleration_5'].abs() - data['volume_acceleration_5'].abs()
    
    # Amount-Weighted Signal Construction
    alpha_values = []
    
    for idx, row in data.iterrows():
        if pd.isna(row['price_momentum_10']) or pd.isna(row['volume_momentum_10']):
            alpha_values.append(np.nan)
            continue
            
        regime = row['regime']
        price_momentum = row['price_momentum_10']
        volume_momentum = row['volume_momentum_10']
        divergence_consistency = row['divergence_consistency']
        acceleration_gap = row['acceleration_gap'] if not pd.isna(row['acceleration_gap']) else 0
        
        if regime == 'strong':
            # Strong Trend: Volume Confirmation 70%, Price Momentum 20%, Acceleration 10%
            volume_weight = 0.7
            price_weight = 0.2
            accel_weight = 0.1
            
        elif regime == 'moderate':
            # Moderate Trend: Balanced 40% each for Price and Volume, Acceleration 20%
            volume_weight = 0.4
            price_weight = 0.4
            accel_weight = 0.2
            
        else:  # weak regime
            # Weak Trend: Price Momentum 70%, Volume Confirmation 20%, Acceleration 10%
            volume_weight = 0.2
            price_weight = 0.7
            accel_weight = 0.1
        
        # Calculate weighted components
        volume_component = volume_weight * volume_momentum * (1 if divergence_consistency < 2 else -1)
        price_component = price_weight * price_momentum
        accel_component = accel_weight * acceleration_gap
        
        # Combine components
        raw_alpha = volume_component + price_component + accel_component
        alpha_values.append(raw_alpha)
    
    # Create alpha series
    alpha_series = pd.Series(alpha_values, index=data.index)
    
    # Apply cross-sectional ranking
    def cross_sectional_rank(series):
        if len(series) < 2:
            return series
        return (series.rank() - 1) / (len(series) - 1) - 0.5
    
    # Group by date and apply cross-sectional ranking
    alpha_final = alpha_series.groupby(alpha_series.index).transform(cross_sectional_rank)
    
    return alpha_final
