import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Multi-Timeframe Momentum
    df = df.copy()
    
    # Very Short-term (2-day) momentum
    df['momentum_2d'] = df['close'].pct_change(periods=2)
    
    # Short-term (5-day) momentum
    df['momentum_5d'] = df['close'].pct_change(periods=5)
    
    # Medium-term (10-day) momentum
    df['momentum_10d'] = df['close'].pct_change(periods=10)
    
    # Long-term (20-day) momentum
    df['momentum_20d'] = df['close'].pct_change(periods=20)
    
    # Assess Momentum Convergence
    momentum_cols = ['momentum_2d', 'momentum_5d', 'momentum_10d', 'momentum_20d']
    
    # Count aligned momentum directions
    df['aligned_count'] = 0
    df['aligned_sum'] = 0.0
    
    for i in range(len(df)):
        if i >= 20:  # Ensure we have enough data for all timeframes
            current_momentums = [df[col].iloc[i] for col in momentum_cols]
            signs = [1 if x > 0 else -1 if x < 0 else 0 for x in current_momentums]
            
            # Count how many have the same sign as the majority
            if len([s for s in signs if s != 0]) > 0:
                majority_sign = 1 if sum(signs) > 0 else -1
                aligned_count = sum(1 for s in signs if s == majority_sign)
                aligned_sum = sum(current_momentums[j] for j in range(len(signs)) if signs[j] == majority_sign)
                
                df.loc[df.index[i], 'aligned_count'] = aligned_count
                df.loc[df.index[i], 'aligned_sum'] = aligned_sum
    
    # Calculate Convergence Strength
    df['convergence_strength'] = df['aligned_sum'] / 4  # Divide by number of timeframes
    
    # Calculate Volume-Weighted Adjustment
    # Identify High Volume Periods
    df['volume_ma_20d'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['high_volume'] = (df['volume'] > df['volume_ma_20d']).astype(int)
    
    # Compute Volume Momentum
    df['volume_momentum'] = df['volume'].pct_change(periods=5)
    
    # Create Volume Multiplier
    df['volume_multiplier'] = 1.0
    for i in range(len(df)):
        if i >= 5:  # Ensure we have enough data for volume momentum
            if df['high_volume'].iloc[i] == 1 and df['volume_momentum'].iloc[i] > 0:
                df.loc[df.index[i], 'volume_multiplier'] = 1.5
            elif df['high_volume'].iloc[i] == 0 and df['volume_momentum'].iloc[i] < 0:
                df.loc[df.index[i], 'volume_multiplier'] = 0.7
    
    # Detect Price-Volume Divergence
    # Normalize momentum and volume series
    df['norm_momentum'] = (df['momentum_5d'] - df['momentum_5d'].rolling(window=20, min_periods=1).mean()) / df['momentum_5d'].rolling(window=20, min_periods=1).std()
    df['norm_volume'] = (df['volume_momentum'] - df['volume_momentum'].rolling(window=20, min_periods=1).mean()) / df['volume_momentum'].rolling(window=20, min_periods=1).std()
    
    # Calculate Divergence Magnitude
    df['divergence_magnitude'] = 0.0
    for i in range(len(df)):
        if i >= 20:  # Ensure we have enough data for normalization
            momentum_val = df['norm_momentum'].iloc[i]
            volume_val = df['norm_volume'].iloc[i]
            
            if not (pd.isna(momentum_val) or pd.isna(volume_val)):
                abs_diff = abs(momentum_val - volume_val)
                
                # Determine direction indicator
                if momentum_val > 0 and volume_val < 0:
                    direction = -1  # Weakness
                elif momentum_val < 0 and volume_val > 0:
                    direction = 1   # Strength
                else:
                    direction = 0
                
                df.loc[df.index[i], 'divergence_magnitude'] = abs_diff * direction
    
    # Generate Composite Alpha Factor
    df['alpha_factor'] = df['convergence_strength'] * df['volume_multiplier'] + df['divergence_magnitude']
    
    # Clean up intermediate columns
    result = df['alpha_factor'].copy()
    
    return result
