import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Intraday Price-Volume Divergence Factor
    Combines price position metrics, volume trend consistency, and divergence detection
    to predict future stock returns based on historical patterns.
    """
    df = data.copy()
    
    # Price Position Metrics
    df['high_to_close_ratio'] = df['high'] / df['close']
    df['low_to_close_ratio'] = df['low'] / df['close']
    df['close_position_range'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Price Range Strength
    df['normalized_range'] = (df['high'] - df['low']) / df['close']
    df['range_3d_avg'] = df['normalized_range'].rolling(window=3, min_periods=1).mean()
    df['range_expansion_ratio'] = df['normalized_range'] / df['range_3d_avg'].replace(0, np.nan)
    
    # Volume Momentum Analysis
    df['volume_acceleration'] = (df['volume'] / df['volume'].shift(2).replace(0, np.nan)) - \
                               (df['volume'].shift(2) / df['volume'].shift(4).replace(0, np.nan))
    df['volume_change_ratio'] = df['volume'] / df['volume'].shift(1).replace(0, np.nan)
    
    # Volume Volatility Component
    df['volume_std_5d'] = df['volume'].rolling(window=5, min_periods=1).std()
    df['volume_stability'] = df['volume'] / (df['volume_std_5d'] + 1e-6)
    
    # Price-Volume Divergence Detection
    df['price_strength'] = df['close_position_range'] * df['range_expansion_ratio']
    df['volume_strength'] = df['volume_acceleration'] * df['volume_stability']
    
    # Divergence signals
    df['strong_price_weak_volume'] = ((df['price_strength'] > df['price_strength'].rolling(window=5, min_periods=1).quantile(0.7)) & 
                                     (df['volume_strength'] < df['volume_strength'].rolling(window=5, min_periods=1).quantile(0.3))).astype(int)
    
    df['weak_price_strong_volume'] = ((df['price_strength'] < df['price_strength'].rolling(window=5, min_periods=1).quantile(0.3)) & 
                                     (df['volume_strength'] > df['volume_strength'].rolling(window=5, min_periods=1).quantile(0.7))).astype(int)
    
    # Trend Persistence Probability
    df['high_volume_continuation'] = ((df['volume_strength'] > df['volume_strength'].rolling(window=5, min_periods=1).quantile(0.7)) & 
                                     (df['price_strength'] > 0)).astype(int)
    
    df['low_volume_reversal'] = ((df['volume_strength'] < df['volume_strength'].rolling(window=5, min_periods=1).quantile(0.3)) & 
                                (df['price_strength'] > 0)).astype(int)
    
    # Historical Pattern Integration
    df['pattern_similarity'] = 0.0
    for i in range(3, len(df)):
        if i >= 5:
            current_pattern = df[['price_strength', 'volume_strength', 'normalized_range']].iloc[i-2:i+1].values.flatten()
            similarities = []
            for j in range(3, i-2):
                historical_pattern = df[['price_strength', 'volume_strength', 'normalized_range']].iloc[j-2:j+1].values.flatten()
                if len(current_pattern) == len(historical_pattern):
                    similarity = 1 / (1 + np.sqrt(np.sum((current_pattern - historical_pattern) ** 2)))
                    similarities.append(similarity)
            if similarities:
                df.loc[df.index[i], 'pattern_similarity'] = np.mean(similarities[-3:]) if len(similarities) >= 3 else np.mean(similarities)
    
    # Final Factor Calculation
    df['divergence_factor'] = (
        df['strong_price_weak_volume'] * -1.0 +  # Negative signal for divergence
        df['weak_price_strong_volume'] * 1.5 +   # Positive signal for potential reversal
        df['high_volume_continuation'] * 0.8 +   # Positive for trend continuation
        df['low_volume_reversal'] * -0.6 +       # Negative for potential reversal
        df['pattern_similarity'] * 0.5           # Weight from historical patterns
    )
    
    # Smooth the factor with rolling mean
    df['final_factor'] = df['divergence_factor'].rolling(window=3, min_periods=1).mean()
    
    return df['final_factor']
