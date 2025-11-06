import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum-Trend Convergence Factor
    Combines momentum, trend persistence, volume validation, and pattern matching
    to generate a robust alpha factor for stock return prediction.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Combined Momentum-Trend Signal
    # Intraday Momentum Component
    data['TrueRange'] = data['high'] - data['low']
    data['MidPrice'] = (data['high'] + data['low']) / 2
    data['DirectionalMomentum'] = np.where(
        data['TrueRange'] > 0,
        (data['close'] - data['MidPrice']) / data['TrueRange'],
        0
    )
    
    # Momentum Strength Assessment
    data['AbsMomentum'] = np.abs(data['DirectionalMomentum'])
    data['MomentumPersistence'] = data['DirectionalMomentum'].rolling(window=5, min_periods=3).mean()
    
    # Trend Persistence Component
    data['HighToClose'] = data['high'] / data['close']
    data['LowToClose'] = data['low'] / data['close']
    data['NormalizedRange'] = (data['high'] - data['low']) / data['close']
    data['ClosePosition'] = np.where(
        (data['high'] - data['low']) > 0,
        (data['close'] - data['low']) / (data['high'] - data['low']),
        0.5
    )
    
    # Trend Strength Metrics
    data['TrendStrength'] = (
        data['HighToClose'].rolling(window=5).std() + 
        data['LowToClose'].rolling(window=5).std() +
        data['NormalizedRange'].rolling(window=5).mean()
    )
    
    # 2. Volume-Enhanced Signal Validation
    # Volume Profile Analysis
    data['VolumeMA5'] = data['volume'].rolling(window=5).mean()
    data['VolumeConcentration'] = data['volume'] / data['VolumeMA5']
    
    # Volume-Trend Alignment
    data['VolumeChangeRatio'] = data['volume'] / data['volume'].shift(1)
    data['VolumeChangeRatio'] = data['VolumeChangeRatio'].fillna(1)
    
    # Volume-Price Trend Consistency
    data['VolumePriceCorr'] = data['volume'].rolling(window=10).corr(data['close'])
    data['VolumePriceCorr'] = data['VolumePriceCorr'].fillna(0)
    
    # Momentum-Volume Divergence Detection
    data['MomentumVolumeDivergence'] = (
        np.sign(data['DirectionalMomentum']) * 
        np.sign(data['VolumeChangeRatio'] - 1) * 
        np.abs(data['DirectionalMomentum'])
    )
    
    # 3. Pattern-Based Signal Refinement
    # Multi-feature Pattern Similarity with dynamic lookback
    def calculate_pattern_similarity(row_idx, data, lookback=5):
        if row_idx < lookback:
            return 0
        
        current_features = data.iloc[row_idx][[
            'DirectionalMomentum', 'NormalizedRange', 'ClosePosition', 'VolumeConcentration'
        ]].values
        
        similarities = []
        for i in range(1, min(lookback + 1, row_idx)):
            past_features = data.iloc[row_idx - i][[
                'DirectionalMomentum', 'NormalizedRange', 'ClosePosition', 'VolumeConcentration'
            ]].values
            
            # Calculate cosine similarity
            dot_product = np.dot(current_features, past_features)
            norm_current = np.linalg.norm(current_features)
            norm_past = np.linalg.norm(past_features)
            
            if norm_current > 0 and norm_past > 0:
                similarity = dot_product / (norm_current * norm_past)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0
    
    # Calculate pattern similarity for each row
    pattern_similarities = []
    for i in range(len(data)):
        similarity = calculate_pattern_similarity(i, data, lookback=7)
        pattern_similarities.append(similarity)
    
    data['PatternSimilarity'] = pattern_similarities
    
    # 4. Signal Convergence Scoring
    # Momentum-Trend Alignment Score
    data['MomentumTrendAlignment'] = (
        data['DirectionalMomentum'] * 
        (data['ClosePosition'] - 0.5) * 
        data['TrendStrength']
    )
    
    # Volume Confirmation Strength
    data['VolumeConfirmation'] = (
        data['VolumeConcentration'] * 
        np.abs(data['VolumePriceCorr']) * 
        np.where(data['MomentumVolumeDivergence'] > 0, 1, -1)
    )
    
    # Final Factor Calculation
    data['IntradayMomentumTrendFactor'] = (
        data['MomentumTrendAlignment'] * 0.4 +
        data['VolumeConfirmation'] * 0.3 +
        data['PatternSimilarity'] * 0.2 +
        data['MomentumPersistence'] * 0.1
    )
    
    # Normalize the final factor
    factor_mean = data['IntradayMomentumTrendFactor'].rolling(window=20, min_periods=10).mean()
    factor_std = data['IntradayMomentumTrendFactor'].rolling(window=20, min_periods=10).std()
    data['NormalizedFactor'] = np.where(
        factor_std > 0,
        (data['IntradayMomentumTrendFactor'] - factor_mean) / factor_std,
        0
    )
    
    return data['NormalizedFactor']
