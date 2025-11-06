import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Trend Components
    data['high_to_close_ratio'] = data['high'] / data['close']
    data['low_to_close_ratio'] = data['low'] / data['close']
    data['close_position_range'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['close_position_range'] = data['close_position_range'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Integration
    data['volume_change_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_change_ratio'] = data['volume_change_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Price Alignment Check
    data['price_change'] = data['close'].pct_change()
    data['volume_price_alignment'] = np.sign(data['price_change']) * np.sign(data['volume_change_ratio'])
    
    # Pattern Recognition - Historical Pattern Matching
    pattern_window = 5
    data['pattern_similarity'] = 0.0
    
    for i in range(pattern_window, len(data)):
        current_pattern = data['close'].iloc[i-pattern_window:i].pct_change().dropna().values
        
        if len(current_pattern) == pattern_window - 1:
            similarities = []
            for j in range(max(3, i-10), i-pattern_window):
                if j >= 0:
                    historical_pattern = data['close'].iloc[j:j+pattern_window].pct_change().dropna().values
                    if len(historical_pattern) == pattern_window - 1:
                        # Calculate cosine similarity
                        dot_product = np.dot(current_pattern, historical_pattern)
                        norm_current = np.linalg.norm(current_pattern)
                        norm_historical = np.linalg.norm(historical_pattern)
                        
                        if norm_current > 0 and norm_historical > 0:
                            similarity = dot_product / (norm_current * norm_historical)
                            similarities.append((similarity, data['close'].iloc[j+pattern_window] / data['close'].iloc[j+pattern_window-1] - 1))
            
            if similarities:
                weights = np.array([abs(sim) for sim, _ in similarities])
                returns = np.array([ret for _, ret in similarities])
                
                if np.sum(weights) > 0:
                    data.loc[data.index[i], 'pattern_similarity'] = np.average(returns, weights=weights)
    
    # Combine components into final factor
    data['intraday_trend_strength'] = (
        (data['high_to_close_ratio'] - 1) * 2 +  # Strength from high proximity
        (1 - data['low_to_close_ratio']) * 2 +   # Strength from low distance
        (data['close_position_range'] - 0.5) * 4  # Position in daily range
    )
    
    # Volume confirmation
    data['volume_confirmation'] = data['volume_price_alignment'] * np.log1p(abs(data['volume_change_ratio']))
    
    # Final factor combining trend strength, volume confirmation, and pattern prediction
    factor = (
        data['intraday_trend_strength'] * 
        (1 + data['volume_confirmation']) * 
        (1 + data['pattern_similarity'])
    )
    
    return factor
