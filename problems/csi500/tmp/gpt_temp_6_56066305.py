import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Convergence Alpha Factor
    
    This factor combines multi-timeframe momentum analysis with volume confirmation
    and adapts weighting based on market participation and volatility regimes.
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Analysis
    # Price Momentum Components
    price_momentum_5d = data['close'].pct_change(5)
    price_momentum_10d = data['close'].pct_change(10)
    price_momentum_20d = data['close'].pct_change(20)
    price_momentum_60d = data['close'].pct_change(60)
    
    # Volume Momentum Components
    volume_momentum_5d = data['volume'].pct_change(5)
    volume_momentum_10d = data['volume'].pct_change(10)
    volume_momentum_20d = data['volume'].pct_change(20)
    volume_momentum_60d = data['volume'].pct_change(60)
    
    # Signal Smoothing and Acceleration
    # Exponential Smoothing (alpha=0.3)
    alpha_smooth = 0.3
    
    smoothed_price_5d = price_momentum_5d.ewm(alpha=alpha_smooth).mean()
    smoothed_price_10d = price_momentum_10d.ewm(alpha=alpha_smooth).mean()
    smoothed_price_20d = price_momentum_20d.ewm(alpha=alpha_smooth).mean()
    smoothed_price_60d = price_momentum_60d.ewm(alpha=alpha_smooth).mean()
    
    smoothed_volume_5d = volume_momentum_5d.ewm(alpha=alpha_smooth).mean()
    smoothed_volume_10d = volume_momentum_10d.ewm(alpha=alpha_smooth).mean()
    smoothed_volume_20d = volume_momentum_20d.ewm(alpha=alpha_smooth).mean()
    smoothed_volume_60d = volume_momentum_60d.ewm(alpha=alpha_smooth).mean()
    
    # Momentum Acceleration Calculation
    price_acceleration = smoothed_price_5d - smoothed_price_10d
    volume_acceleration = smoothed_volume_5d - smoothed_volume_10d
    
    # Amount-Based Regime Detection
    amount_20d_avg = data['amount'].rolling(window=20).mean()
    amount_acceleration = data['amount'].pct_change(5)
    
    # Amount persistence (count of days above 20-day average in last 5 days)
    amount_above_avg = (data['amount'] > amount_20d_avg.shift(1)).rolling(window=5).sum()
    
    # Amount volatility
    amount_volatility = data['amount'].rolling(window=20).std() / amount_20d_avg
    
    # Regime Classification - Amount Participation
    high_participation = data['amount'] > (1.2 * amount_20d_avg)
    low_participation = data['amount'] < (0.8 * amount_20d_avg)
    normal_participation = ~(high_participation | low_participation)
    
    # Volatility Regime Assessment
    daily_range = (data['high'] - data['low']) / data['close']
    range_20d_avg = daily_range.rolling(window=20).mean()
    range_5d_avg = daily_range.rolling(window=5).mean()
    
    # Range persistence (count of days above 20-day average in last 5 days)
    range_above_avg = (daily_range > range_20d_avg.shift(1)).rolling(window=5).sum()
    
    # Range acceleration
    range_acceleration = daily_range / range_5d_avg
    
    # Volatility Classification
    high_volatility = daily_range > (1.3 * range_20d_avg)
    low_volatility = daily_range < (0.7 * range_20d_avg)
    normal_volatility = ~(high_volatility | low_volatility)
    
    # Convergence-Divergence Signal Construction
    # Multi-timeframe Alignment
    short_term_convergence = smoothed_price_5d * smoothed_volume_5d
    medium_term_alignment = smoothed_price_10d * smoothed_volume_10d
    long_term_confirmation = smoothed_price_20d * smoothed_volume_20d
    ultra_long_term = smoothed_price_60d * smoothed_volume_60d
    
    # Signal Strength Assessment
    # Consistency score across timeframes
    momentum_signs = pd.DataFrame({
        'short': np.sign(smoothed_price_5d),
        'medium': np.sign(smoothed_price_10d),
        'long': np.sign(smoothed_price_20d)
    })
    consistency_score = momentum_signs.sum(axis=1) / 3.0
    
    # Acceleration alignment
    acceleration_alignment = np.sign(price_acceleration) * np.sign(volume_acceleration)
    
    # Regime-Adaptive Weighting Scheme
    composite_scores = pd.Series(index=data.index, dtype=float)
    
    for idx in data.index:
        # Base weights for different timeframes
        base_weights = {
            'short': 0.25,
            'medium': 0.35,
            'long': 0.30,
            'ultra_long': 0.10
        }
        
        # Adjust weights based on regimes
        if high_volatility.loc[idx]:
            # High Volatility Regime
            base_weights['short'] *= 0.8  # Reduce short-term weight
            base_weights['medium'] *= 1.2  # Emphasize medium-term
            base_weights['ultra_long'] *= 0.5  # Reduce ultra-long-term
            volume_weight_factor = 0.7  # Emphasize volume confirmation
            
        elif low_volatility.loc[idx]:
            # Low Volatility Regime
            base_weights['long'] *= 1.3  # Emphasize long-term
            base_weights['ultra_long'] *= 1.1  # Include ultra-long-term
            base_weights['short'] *= 0.7  # Reduce short-term sensitivity
            volume_weight_factor = 0.3  # Reduce volume weight
            
        else:
            # Normal Volatility Regime
            volume_weight_factor = 0.5  # Balanced weighting
        
        if high_participation.loc[idx]:
            # High Participation Regime
            base_weights['medium'] *= 1.2  # Emphasize medium-term
            acceleration_weight = 0.3  # Higher weight for acceleration
            
        elif low_participation.loc[idx]:
            # Low Participation Regime
            base_weights['short'] *= 0.6  # Conservative short-term
            base_weights['medium'] *= 1.1  # Emphasize medium-term confirmation
            base_weights['long'] *= 1.2  # Emphasize long-term confirmation
            acceleration_weight = 0.1  # Lower weight for acceleration
            
        else:
            # Normal Participation Regime
            acceleration_weight = 0.2  # Moderate acceleration weight
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v/total_weight for k, v in base_weights.items()}
        
        # Calculate composite score with regime-adaptive weighting
        price_component = (
            normalized_weights['short'] * smoothed_price_5d.loc[idx] +
            normalized_weights['medium'] * smoothed_price_10d.loc[idx] +
            normalized_weights['long'] * smoothed_price_20d.loc[idx] +
            normalized_weights['ultra_long'] * smoothed_price_60d.loc[idx]
        )
        
        volume_component = (
            normalized_weights['short'] * smoothed_volume_5d.loc[idx] +
            normalized_weights['medium'] * smoothed_volume_10d.loc[idx] +
            normalized_weights['long'] * smoothed_volume_20d.loc[idx] +
            normalized_weights['ultra_long'] * smoothed_volume_60d.loc[idx]
        )
        
        # Apply volume weight factor based on regime
        regime_weighted_score = (
            (1 - volume_weight_factor) * price_component +
            volume_weight_factor * volume_component
        )
        
        # Add acceleration component
        acceleration_component = acceleration_weight * acceleration_alignment.loc[idx]
        
        # Final composite score
        composite_scores.loc[idx] = (
            regime_weighted_score + 
            acceleration_component +
            0.1 * consistency_score.loc[idx]  # Small consistency bonus
        )
    
    # Cross-Sectional Ranking Enhancement
    # Calculate rolling z-score within regime groups
    final_factor = pd.Series(index=data.index, dtype=float)
    
    for regime_vol in ['high_vol', 'low_vol', 'normal_vol']:
        for regime_part in ['high_part', 'low_part', 'normal_part']:
            if regime_vol == 'high_vol':
                vol_mask = high_volatility
            elif regime_vol == 'low_vol':
                vol_mask = low_volatility
            else:
                vol_mask = normal_volatility
                
            if regime_part == 'high_part':
                part_mask = high_participation
            elif regime_part == 'low_part':
                part_mask = low_participation
            else:
                part_mask = normal_participation
                
            regime_mask = vol_mask & part_mask
            
            if regime_mask.any():
                regime_scores = composite_scores[regime_mask]
                # Calculate z-score within regime
                regime_mean = regime_scores.rolling(window=20, min_periods=10).mean()
                regime_std = regime_scores.rolling(window=20, min_periods=10).std()
                z_scores = (regime_scores - regime_mean) / regime_std
                final_factor[regime_mask] = z_scores[regime_mask]
    
    # Fill any remaining NaN values with raw composite scores
    final_factor = final_factor.fillna(composite_scores)
    
    return final_factor
