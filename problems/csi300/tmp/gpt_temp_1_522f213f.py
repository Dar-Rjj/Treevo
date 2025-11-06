import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Range-Volume Divergence Alpha Factor
    """
    data = df.copy()
    
    # Daily Price Range
    data['daily_range'] = data['high'] - data['low']
    
    # Volatility Regime Classification
    data['volatility_20d'] = data['daily_range'].rolling(window=20).std()
    data['volatility_median_60d'] = data['volatility_20d'].rolling(window=60).median()
    data['high_vol_regime'] = data['volatility_20d'] > data['volatility_median_60d']
    
    # Multi-Period Range Momentum Calculation
    # Price Range Momentum
    data['range_momentum_3d'] = data['daily_range'] / data['daily_range'].shift(3) - 1
    data['range_momentum_5d'] = data['daily_range'] / data['daily_range'].shift(5) - 1
    data['range_momentum_10d'] = data['daily_range'] / data['daily_range'].shift(10) - 1
    
    # Close Price Momentum
    data['close_momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['close_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['close_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Volume Momentum Components
    data['volume_momentum_3d'] = data['volume'] / data['volume'].shift(3)
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5)
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10)
    
    # Initialize divergence scores
    divergence_scores = pd.Series(index=data.index, dtype=float)
    
    # Calculate regime-specific divergence
    for i in range(len(data)):
        if i < 20:  # Ensure enough data for calculations
            divergence_scores.iloc[i] = 0
            continue
            
        current_data = data.iloc[i]
        is_high_vol = current_data['high_vol_regime']
        
        if is_high_vol:
            # High Volatility Regime: Range-Volume Alignment
            range_momentum_3d = current_data['range_momentum_3d']
            volume_momentum_3d = current_data['volume_momentum_3d']
            
            if not np.isnan(range_momentum_3d) and not np.isnan(volume_momentum_3d):
                # Check alignment direction
                if (range_momentum_3d > 0 and volume_momentum_3d > 1) or (range_momentum_3d < 0 and volume_momentum_3d < 1):
                    alignment_score = 1
                else:
                    alignment_score = -1
                
                # Weight by volume momentum magnitude
                volume_weight = np.log(volume_momentum_3d) if volume_momentum_3d > 0 else -1
                divergence_scores.iloc[i] = alignment_score * volume_weight
            else:
                divergence_scores.iloc[i] = 0
                
        else:
            # Low Volatility Regime: Divergence Persistence
            range_momentum_5d = current_data['range_momentum_5d']
            volume_momentum_5d = current_data['volume_momentum_5d']
            
            if not np.isnan(range_momentum_5d) and not np.isnan(volume_momentum_5d):
                # Calculate divergence direction
                if (range_momentum_5d > 0 and volume_momentum_5d < 1) or (range_momentum_5d < 0 and volume_momentum_5d > 1):
                    divergence_direction = -1  # Bearish divergence
                else:
                    divergence_direction = 1   # Bullish convergence
                
                # Calculate persistence (consecutive days with same divergence pattern)
                persistence = 1
                for j in range(1, min(10, i)):
                    prev_idx = i - j
                    prev_range = data.iloc[prev_idx]['range_momentum_5d']
                    prev_volume = data.iloc[prev_idx]['volume_momentum_5d']
                    
                    if np.isnan(prev_range) or np.isnan(prev_volume):
                        break
                        
                    prev_divergence = (prev_range > 0 and prev_volume < 1) or (prev_range < 0 and prev_volume > 1)
                    current_divergence = (range_momentum_5d > 0 and volume_momentum_5d < 1) or (range_momentum_5d < 0 and volume_momentum_5d > 1)
                    
                    if prev_divergence == current_divergence:
                        persistence += 1
                    else:
                        break
                
                # Calculate correlation between 5-day range and volume changes
                if i >= 25:  # Need enough data for correlation
                    recent_range = data['daily_range'].iloc[i-20:i+1].pct_change().dropna()
                    recent_volume = data['volume'].iloc[i-20:i+1].pct_change().dropna()
                    
                    if len(recent_range) > 5 and len(recent_volume) > 5:
                        correlation = recent_range.corr(recent_volume)
                        if np.isnan(correlation):
                            correlation = 0
                    else:
                        correlation = 0
                else:
                    correlation = 0
                
                divergence_scores.iloc[i] = divergence_direction * persistence * (1 + abs(correlation))
            else:
                divergence_scores.iloc[i] = 0
    
    # Apply range-adjusted weighting
    range_adjusted_scores = divergence_scores * data['daily_range']
    
    # Apply momentum decay with exponential weighting
    decay_weights = np.exp(-np.arange(5) / 2)  # Exponential decay for last 5 days
    decay_weights = decay_weights / decay_weights.sum()  # Normalize
    
    final_scores = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 5:
            final_scores.iloc[i] = range_adjusted_scores.iloc[i]
        else:
            recent_scores = range_adjusted_scores.iloc[i-4:i+1].values
            if len(recent_scores) == 5:
                final_scores.iloc[i] = np.dot(recent_scores, decay_weights)
            else:
                final_scores.iloc[i] = range_adjusted_scores.iloc[i]
    
    # Handle NaN values
    final_scores = final_scores.fillna(0)
    
    return final_scores
