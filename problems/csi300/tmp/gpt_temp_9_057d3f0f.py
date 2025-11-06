import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Divergence with Dynamic Persistence Scoring
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Momentum Analysis
    # Ultra-Short Momentum (1-day)
    data['momentum_1d'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Short-Term Momentum (3-day)
    data['momentum_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    
    # Medium-Term Momentum (8-day)
    data['momentum_8d'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    
    # Momentum Divergence Framework
    # Acceleration Divergence
    data['accel_divergence'] = data['momentum_3d'] - data['momentum_8d']
    
    # Reversal Detection
    data['reversal_signal'] = np.where(
        np.sign(data['momentum_1d']) != np.sign(data['momentum_8d']), -1, 1
    )
    
    # Momentum Consistency Score
    def momentum_consistency(row):
        aligned_count = 0
        if np.sign(row['momentum_1d']) == np.sign(row['momentum_8d']):
            aligned_count += 1
        if np.sign(row['momentum_3d']) == np.sign(row['momentum_8d']):
            aligned_count += 2  # Higher weight for medium-term
        return aligned_count
    
    data['momentum_consistency'] = data.apply(momentum_consistency, axis=1)
    
    # Volume-Price Divergence System
    # Volume Trend Analysis
    data['volume_momentum_2d'] = (data['volume'] - data['volume'].shift(2)) / data['volume'].shift(2)
    
    # Volume Persistence
    volume_3d_avg = data['volume'].rolling(window=3).mean()
    data['volume_above_avg'] = data['volume'] > volume_3d_avg
    
    def volume_persistence_score(series, window=5):
        scores = []
        for i in range(len(series)):
            if i < window:
                scores.append(0)
                continue
            window_data = series.iloc[i-window+1:i+1]
            persistence = 0
            for j, val in enumerate(reversed(window_data)):
                if val:
                    persistence += (0.8 ** j)  # Exponential decay
            scores.append(persistence)
        return pd.Series(scores, index=series.index)
    
    data['volume_persistence'] = volume_persistence_score(data['volume_above_avg'])
    
    # Volume Strength Score
    data['volume_strength'] = (
        0.6 * data['volume_momentum_2d'].fillna(0) + 
        0.4 * data['volume_persistence'].fillna(0)
    )
    
    # Divergence Detection Matrix
    def volume_confidence_multiplier(row):
        momentum_8d_sign = np.sign(row['momentum_8d'])
        volume_strength_sign = np.sign(row['volume_strength'])
        
        if momentum_8d_sign == volume_strength_sign:
            return 1.5  # Strong Confirmation
        elif momentum_8d_sign == 0 or volume_strength_sign == 0:
            return 1.0  # Neutral
        elif abs(row['momentum_8d']) < 0.02 and abs(row['volume_strength']) < 0.1:
            return 0.7  # Mild Divergence
        else:
            return 0.3  # Strong Divergence
    
    data['volume_confidence'] = data.apply(volume_confidence_multiplier, axis=1)
    
    # Adaptive Persistence Scoring
    # Direction Persistence Engine
    def direction_alignment_score(row):
        score = 0
        if np.sign(row['momentum_1d']) == np.sign(row['momentum_8d']):
            score += 1
        if np.sign(row['momentum_3d']) == np.sign(row['momentum_8d']):
            score += 2
        return score / 3.0  # Normalize to 0-1
    
    data['direction_alignment'] = data.apply(direction_alignment_score, axis=1)
    
    # Exponential Persistence Decay
    def calculate_persistence(series, decay_factor=0.85, window=10):
        persistence_scores = []
        for i in range(len(series)):
            if i < window:
                persistence_scores.append(0)
                continue
            
            current_sign = np.sign(series.iloc[i])
            persistence = 0
            for j in range(window):
                idx = i - j
                if np.sign(series.iloc[idx]) == current_sign:
                    persistence += decay_factor ** j
            persistence_scores.append(persistence)
        return pd.Series(persistence_scores, index=series.index)
    
    data['momentum_persistence'] = calculate_persistence(data['momentum_8d'])
    
    # Direction Persistence Score
    data['direction_persistence'] = (
        data['direction_alignment'] * data['momentum_persistence']
    )
    
    # Magnitude Stability Assessment
    data['momentum_volatility'] = data['momentum_3d'].rolling(window=6).std()
    data['momentum_8d_6d_avg'] = data['momentum_8d'].rolling(window=6).mean()
    data['magnitude_consistency'] = (
        data['momentum_8d'] / data['momentum_8d_6d_avg']
    ).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Stability Score
    data['stability_score'] = (
        (1 / (1 + data['momentum_volatility'].fillna(0))) * 
        (1 / (1 + abs(data['magnitude_consistency'] - 1)))
    )
    
    # Dynamic Persistence Metric
    data['dynamic_persistence'] = (
        data['direction_persistence'] * 
        data['stability_score'] * 
        abs(data['momentum_8d'])
    )
    
    # Price Action Context Integration
    # Intraday Range Analysis
    data['range_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, 1e-10)
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1e-10)
    
    def range_quality_adjustment(row):
        momentum_sign = np.sign(row['momentum_8d'])
        range_pos = row['range_position']
        
        if momentum_sign > 0 and range_pos > 0.7:
            return 1.2  # Strong bullish quality
        elif momentum_sign < 0 and range_pos < 0.3:
            return 1.2  # Strong bearish quality
        elif 0.4 <= range_pos <= 0.6:
            return 1.0  # Medium quality
        else:
            return 0.8  # Weak quality
    
    data['range_quality'] = data.apply(range_quality_adjustment, axis=1)
    
    # Volatility Normalization
    data['daily_volatility'] = (data['high'] - data['low']) / data['close']
    data['volatility_5d_avg'] = data['daily_volatility'].rolling(window=5).mean()
    
    # Amount-Based Liquidity Filter
    data['amount_5d_avg'] = data['amount'].rolling(window=5).mean()
    data['relative_amount'] = data['amount'] / data['amount_5d_avg']
    
    def liquidity_confidence(row):
        rel_amount = row['relative_amount']
        if rel_amount > 1.2:
            return 1.1  # High confidence
        elif 0.8 <= rel_amount <= 1.2:
            return 1.0  # Medium confidence
        else:
            return 0.9  # Low confidence
    
    data['liquidity_weight'] = data.apply(liquidity_confidence, axis=1)
    
    # Composite Alpha Generation
    # Core Momentum Foundation
    core_momentum = data['momentum_8d']
    
    # Volume-Adapted Enhancement
    volume_adapted = core_momentum * data['volume_confidence']
    
    # Persistence Amplification
    persistence_amplified = volume_adapted * data['dynamic_persistence']
    
    # Context-Optimized Final Factor
    final_factor = (
        persistence_amplified * 
        data['range_quality'] * 
        data['liquidity_weight'] / 
        data['daily_volatility'].replace(0, 1e-10)
    )
    
    return final_factor
