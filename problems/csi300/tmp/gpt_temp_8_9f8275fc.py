import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Intraday Price Momentum
    data['high_to_close_momentum'] = (data['high'] / data['close']) - 1
    data['low_to_close_momentum'] = (data['low'] / data['close']) - 1
    data['relative_range'] = (data['high'] - data['low']) / data['close']
    data['close_position_ratio'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['close_position_ratio'] = data['close_position_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    # Calculate Volume Momentum Patterns
    data['volume_momentum_ratio'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) / (data['volume'].shift(1) / data['volume'].shift(2)) - 1
    data['volume_momentum_ratio'] = data['volume_momentum_ratio'].fillna(0)
    data['volume_acceleration'] = data['volume_acceleration'].fillna(0)
    
    # Non-Linear Trend-Volume Interaction
    data['price_volume_momentum_product'] = data['high_to_close_momentum'] * data['volume_momentum_ratio']
    data['range_volume_interaction'] = data['relative_range'] * data['volume_acceleration']
    
    # Directional weighting components
    data['trend_direction'] = np.where(data['close'] > data['open'], 1, -1)
    data['signed_price_volume'] = data['trend_direction'] * data['price_volume_momentum_product']
    data['position_adjusted_volume'] = data['close_position_ratio'] * data['volume_momentum_ratio']
    
    # Calculate base factor
    data['base_factor'] = (
        data['signed_price_volume'] + 
        data['range_volume_interaction'] + 
        data['position_adjusted_volume']
    )
    
    # Historical Pattern Validation with dynamic lookback
    factor_values = []
    
    for i in range(len(data)):
        if i < 10:  # Need enough history for pattern matching
            factor_values.append(0)
            continue
            
        current_features = {
            'high_to_close_momentum': data['high_to_close_momentum'].iloc[i],
            'low_to_close_momentum': data['low_to_close_momentum'].iloc[i],
            'relative_range': data['relative_range'].iloc[i],
            'close_position_ratio': data['close_position_ratio'].iloc[i],
            'volume_momentum_ratio': data['volume_momentum_ratio'].iloc[i],
            'volume_acceleration': data['volume_acceleration'].iloc[i]
        }
        
        # Dynamic lookback period (3-10 days)
        lookback = min(10, i)
        similarities = []
        pattern_returns = []
        volume_consistencies = []
        
        for j in range(max(0, i - lookback), i):
            # Calculate pattern similarity using Euclidean distance on normalized features
            hist_features = {
                'high_to_close_momentum': data['high_to_close_momentum'].iloc[j],
                'low_to_close_momentum': data['low_to_close_momentum'].iloc[j],
                'relative_range': data['relative_range'].iloc[j],
                'close_position_ratio': data['close_position_ratio'].iloc[j],
                'volume_momentum_ratio': data['volume_momentum_ratio'].iloc[j],
                'volume_acceleration': data['volume_acceleration'].iloc[j]
            }
            
            # Normalize features for similarity calculation
            current_norm = np.array(list(current_features.values()))
            hist_norm = np.array(list(hist_features.values()))
            
            # Avoid division by zero in normalization
            current_norm = (current_norm - np.mean(current_norm)) / (np.std(current_norm) + 1e-8)
            hist_norm = (hist_norm - np.mean(hist_norm)) / (np.std(hist_norm) + 1e-8)
            
            similarity = 1 / (1 + np.sqrt(np.sum((current_norm - hist_norm) ** 2)))
            similarities.append(similarity)
            
            # Calculate forward return for historical pattern (using next day's return)
            if j + 1 < len(data):
                forward_return = (data['close'].iloc[j + 1] / data['close'].iloc[j]) - 1
                pattern_returns.append(forward_return)
            else:
                pattern_returns.append(0)
            
            # Volume consistency check
            volume_consistency = 1 - abs(data['volume_momentum_ratio'].iloc[j])
            volume_consistencies.append(volume_consistency)
        
        # Weight predictions by pattern similarity and volume consistency
        if similarities and pattern_returns:
            weights = np.array(similarities) * np.array(volume_consistencies)
            if np.sum(weights) > 0:
                weighted_return = np.average(pattern_returns, weights=weights)
            else:
                weighted_return = 0
        else:
            weighted_return = 0
        
        # Combine base factor with historical pattern prediction
        final_factor = data['base_factor'].iloc[i] + weighted_return
        factor_values.append(final_factor)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=data.index, name='intraday_trend_volume_divergence')
    
    return factor_series
