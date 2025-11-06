import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily range
    daily_range = df['high'] - df['low']
    
    # Volatility Regime Detection
    vol_20d = daily_range.rolling(window=20, min_periods=10).std()
    vol_60d_median = daily_range.rolling(window=60, min_periods=30).median()
    volatility_regime = (vol_20d > vol_60d_median).astype(int)
    
    # Multi-Period Momentum Analysis
    # Range momentum
    range_momentum_3d = daily_range.pct_change(periods=3)
    range_momentum_5d = daily_range.pct_change(periods=5)
    range_momentum_10d = daily_range.pct_change(periods=10)
    
    # Volume momentum
    volume_momentum_3d = df['volume'].pct_change(periods=3)
    volume_momentum_5d = df['volume'].pct_change(periods=5)
    volume_momentum_10d = df['volume'].pct_change(periods=10)
    
    # Combined momentum scores
    range_momentum_combined = (range_momentum_3d + range_momentum_5d + range_momentum_10d) / 3
    volume_momentum_combined = (volume_momentum_3d + volume_momentum_5d + volume_momentum_10d) / 3
    
    # Regime-Specific Divergence Scoring
    divergence_scores = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 10:  # Need enough data for calculations
            divergence_scores.iloc[i] = 0
            continue
            
        current_regime = volatility_regime.iloc[i]
        current_range_momentum = range_momentum_combined.iloc[i]
        current_volume_momentum = volume_momentum_combined.iloc[i]
        
        if current_regime == 1:  # High volatility regime
            # Direction alignment: same sign = positive, different sign = negative
            direction_alignment = np.sign(current_range_momentum * current_volume_momentum)
            # Score = direction alignment × volume momentum magnitude
            score = direction_alignment * abs(current_volume_momentum)
            
        else:  # Low volatility regime
            # Calculate range-volume correlation over recent period
            lookback = min(20, i)
            recent_range = daily_range.iloc[i-lookback:i+1]
            recent_volume = df['volume'].iloc[i-lookback:i+1]
            
            if len(recent_range) >= 5:
                correlation = recent_range.corr(recent_volume)
                correlation = 0 if pd.isna(correlation) else correlation
            else:
                correlation = 0
            
            # Count consecutive days with consistent divergence
            divergence_direction = np.sign(current_range_momentum * current_volume_momentum)
            consecutive_count = 1
            for j in range(1, min(10, i+1)):
                if i-j < 0:
                    break
                prev_range_momentum = range_momentum_combined.iloc[i-j]
                prev_volume_momentum = volume_momentum_combined.iloc[i-j]
                prev_direction = np.sign(prev_range_momentum * prev_volume_momentum)
                
                if prev_direction == divergence_direction:
                    consecutive_count += 1
                else:
                    break
            
            score = consecutive_count * correlation
        
        divergence_scores.iloc[i] = score
    
    # Final Alpha Construction
    # Apply current daily range as scaling factor
    scaled_scores = divergence_scores * daily_range
    
    # Weight recent signals with exponential decay (half-life = 5 days)
    weights = np.exp(-np.arange(len(scaled_scores)) / 5)
    weights = weights[::-1]  # Reverse for proper time weighting
    
    # Apply exponential weighting
    final_alpha = pd.Series(index=df.index, dtype=float)
    for i in range(len(scaled_scores)):
        if i == 0:
            final_alpha.iloc[i] = scaled_scores.iloc[i]
        else:
            # Use available data points with exponential decay
            available_weights = weights[-i-1:]
            available_scores = scaled_scores.iloc[:i+1]
            weighted_avg = np.average(available_scores, weights=available_weights)
            final_alpha.iloc[i] = weighted_avg
    
    return final_alpha
